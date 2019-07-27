package czzhash

import "C"
import (
	"fmt"
	"github.com/Gustav-Simonsson/go-opencl/cl"
	"github.com/classzz/classzz/wire"
	"math"
	"math/big"
	mrand "math/rand"
	"sync"
	"sync/atomic"
	"unsafe"
)

type OpenCLDevice struct {
	deviceId int
	device   *cl.Device
	openCL11 bool // OpenCL version 1.1 and 1.2 are handled a bit different
	openCL12 bool

	binBuf       *cl.MemObject // classzz full Bin in device mem
	headerBuf    *cl.MemObject // Hash of block-to-mine in device mem
	searchBuffer *cl.MemObject

	searchKernel *cl.Kernel

	queue         *cl.CommandQueue
	ctx           *cl.Context
	workGroupSize int

	nonceRand *mrand.Rand
	result    Hash
}

type OpenCLMiner struct {
	mu sync.Mutex

	czzhash *CzzHash // classzz full Bin & cache in host mem

	deviceIds []int
	devices   []*OpenCLDevice

	binSize uint64

	hashRate int32 // Go atomics & uint64 have some issues; int32 is supported on all platforms
}

type pendingSearch struct {
	bufIndex   uint32
	startNonce uint64
}

const (
	// See [4]
	workGroupSize = 32 // must be multiple of 8
)

func NewCL(deviceIds []int) *OpenCLMiner {
	ids := make([]int, len(deviceIds))
	copy(ids, deviceIds)
	return &OpenCLMiner{
		czzhash:   New(),
		binSize:   TBLSize, // to see if we need to update Bin.
		deviceIds: ids,
	}
}

// See [2]. We basically do the same here, but the Go OpenCL bindings
// are at a slightly higher abtraction level.
func InitCL(blockNum uint64, c *OpenCLMiner) error {
	platforms, err := cl.GetPlatforms()
	if err != nil {
		return fmt.Errorf("Plaform error: %v\nCheck your OpenCL installation and then run geth gpuinfo", err)
	}

	var devices []*cl.Device
	for _, p := range platforms {
		ds, err := cl.GetDevices(p, cl.DeviceTypeGPU)
		if err != nil {
			return fmt.Errorf("Devices error: %v\nCheck your GPU drivers and then run geth gpuinfo", err)
		}
		for _, d := range ds {
			devices = append(devices, d)
		}
	}

	pow := New()
	pow.Csatable = pow.GetBin(blockNum) // generates Bin if we don't have it
	c.czzhash = pow

	for _, id := range c.deviceIds {
		if id > len(devices)-1 {
			return fmt.Errorf("Device id not found. See available device ids with: geth gpuinfo")
		} else {
			err := initCLDevice(id, devices[id], c)
			fmt.Println("err:", err)
			if err != nil {
				return err
			}
		}
	}
	if len(c.devices) == 0 {
		return fmt.Errorf("No GPU devices found")
	}
	return nil
}

func initCLDevice(deviceId int, device *cl.Device, c *OpenCLMiner) error {
	devMaxAlloc := uint64(device.MaxMemAllocSize())
	devGlobalMem := uint64(device.GlobalMemSize())

	if device.Version() == "OpenCL 1.0" {
		fmt.Println("Device OpenCL version not supported: ", device.Version())
		return fmt.Errorf("opencl version not supported")
	}
	fmt.Println(device.Version())
	var cl11, cl12 bool
	if device.Version() == "OpenCL 1.1" {
		cl11 = true
	}
	if device.Version() == "OpenCL 1.2" {
		cl12 = true
	}

	// log warnings but carry on; some device drivers report inaccurate values
	if c.binSize > devGlobalMem {
		fmt.Printf("WARNING: device memory may be insufficient: %v. Bin size: %v.\n", devGlobalMem, c.binSize)
	}

	if c.binSize > devMaxAlloc {
		fmt.Printf("WARNING: Bin size (%v) larger than device max memory allocation size (%v).\n", c.binSize, devMaxAlloc)
		fmt.Printf("You probably have to export GPU_MAX_ALLOC_PERCENT=95\n")
	}

	fmt.Printf("Initialising device %v: %v\n", deviceId, device.Name())

	context, err := cl.CreateContext([]*cl.Device{device})
	if err != nil {
		return fmt.Errorf("failed creating context: %v", err)
	}

	queue, err := context.CreateCommandQueue(device, 0)
	if err != nil {
		return fmt.Errorf("command queue err: %v", err)
	}

	// See [4] section 3.2 and [3] "clBuildProgram".
	// The OpenCL kernel code is compiled at run-time.
	program, err := context.CreateProgramWithSource([]string{Kernel})
	if err != nil {
		return fmt.Errorf("program err: %v", err)
	}

	/* if using AMD OpenCL impl, you can set this to debug on x86 CPU device.
	   see AMD OpenCL programming guide section 4.2

	   export in shell before running:
	   export AMD_OCL_BUILD_OPTIONS_APPEND="-g -O0"
	   export CPU_MAX_COMPUTE_UNITS=1

	buildOpts := "-g -cl-opt-disable"

	*/
	buildOpts := ""
	err = program.BuildProgram([]*cl.Device{device}, buildOpts)
	if err != nil {
		return fmt.Errorf("program build err: %v", err)
	}

	var searchKernelName string
	searchKernelName = "czzhash_search"

	searchKernel, err := program.CreateKernel(searchKernelName)
	if err != nil {
		return fmt.Errorf("kernel err: %v", err)
	}

	// (context.go) to work with uint64 as size_t
	if c.binSize > math.MaxInt32 {
		fmt.Println("Bin too large for allocation.")
		return fmt.Errorf("Bin too large for alloc")
	}

	binBuf, err := context.CreateEmptyBuffer(cl.MemReadOnly, TBLSize)
	if err != nil {
		return fmt.Errorf("allocating Bin buf failed: %v", err)
	}

	searchBuffer, err := context.CreateEmptyBuffer(cl.MemWriteOnly, 32)
	if err != nil {
		return fmt.Errorf("allocating Bin buf failed: %v", err)
	}

	//write Bin to device mem
	csa := c.czzhash.Full.Csatable.Bytes()
	csatable := [TBLSize]byte{}
	for i, j := range csa {
		csatable[i] = j
	}

	_, err = queue.EnqueueWriteBuffer(binBuf, true, 0, TBLSize, unsafe.Pointer(&csatable), nil)
	if err != nil {
	}

	deviceStruct := &OpenCLDevice{
		deviceId: deviceId,
		device:   device,
		openCL11: cl11,
		openCL12: cl12,

		binBuf:       binBuf,
		searchBuffer: searchBuffer,

		searchKernel: searchKernel,

		queue: queue,
		ctx:   context,

		workGroupSize: workGroupSize,
	}
	c.devices = append(c.devices, deviceStruct)

	return nil
}

func (c *OpenCLMiner) Search(hash [32]byte, target uint64, stop <-chan struct{}, index int64) (uint64, uint64) {

	headerHash := hash
	fmt.Println("hash:", hash)
	fmt.Println("target:", target)

	d := c.devices[index]

	headerBuf, err := d.ctx.CreateEmptyBuffer(cl.MemReadOnly, 32)

	_, err = d.queue.EnqueueWriteBuffer(headerBuf, true, 0, 32, unsafe.Pointer(&headerHash), nil)
	if err != nil {
		fmt.Println("Error in Search clEnqueueWriterBuffer : ", err)
		return 0, 0
	}

	nonceBuffer, err := d.ctx.CreateEmptyBuffer(cl.MemWriteOnly, 8)
	if err != nil {
		fmt.Println("Error in Search nonceBuffer : ", err)
		return 0, 0
	}

	// wait on this before returning
	var preReturnEvent *cl.Event
	if d.openCL12 {
		preReturnEvent, err = d.ctx.CreateUserEvent()
		if err != nil {
			fmt.Println("Error in Search create CL user event : ", err)
			return 0, 0
		}
	}

	// we grab a single random nonce and sets this as argument to the kernel search function
	// the device will then add each local threads gid to the nonce, creating a unique nonce
	// for each device computing unit executing in parallel
	Nonce, _ := wire.RandomUint64()
	fmt.Println("Nonce:", Nonce)

	err = d.searchKernel.SetArg(1, headerBuf)
	if err != nil {
		fmt.Println("Error in Search clSetKernelArg : ", err)
		return 0, 0
	}

	err = d.searchKernel.SetArg(2, d.binBuf)
	if err != nil {
		fmt.Println("Error in Search clSetKernelArg : ", err)
		return 0, 0
	}

	err = d.searchKernel.SetArg(4, target)
	if err != nil {
		fmt.Println("Error in Search clSetKernelArg : ", err)
		return 0, 0
	}

	isolate := uint32(0xFFFFFFFF)
	err = d.searchKernel.SetArg(5, isolate)
	if err != nil {
		fmt.Println("Error in Search clSetKernelArg : ", err)
		return 0, 0
	}

	for {

		err = d.searchKernel.SetArg(0, d.searchBuffer)
		if err != nil {
			fmt.Println("Error in Search clSetKernelArg : ", err)
			return 0, 0
		}

		err = d.searchKernel.SetArg(6, nonceBuffer)
		if err != nil {
			fmt.Println("Error in Search nonceBuffer : ", err)
			return 0, 0
		}

		err = d.searchKernel.SetArg(3, Nonce)
		if err != nil {
			fmt.Println("Error in Search clSetKernelArg : ", err)
			return 0, 0
		}

		// execute kernel
		_, err = d.queue.EnqueueNDRangeKernel(
			d.searchKernel,
			nil,
			[]int{1},
			[]int{1},
			nil)
		if err != nil {
			fmt.Println("Error in Search clEnqueueNDRangeKernel : ", err)
			return 0, 0
		}

		var result [32]byte
		_, err = d.queue.EnqueueReadBuffer(d.searchBuffer, true, 0, 32, unsafe.Pointer(&result), nil)
		if err != nil {
			fmt.Println("Error in Search clEnqueueMapBuffer: ", err)
			return 0, 0
		}
		fmt.Println("result:", result)

		var stop_nonce uint64
		_, err = d.queue.EnqueueReadBuffer(nonceBuffer, true, 0, 8, unsafe.Pointer(&stop_nonce), nil)
		if err != nil {
			fmt.Println("Error in Search clEnqueueMapBuffer: ", err)
			return 0, 0
		}

		fmt.Println("stop_nonce", stop_nonce)
		if new(big.Int).SetBytes(result[:]).Cmp(big.NewInt(0).SetUint64(target)) <= 0 {
			fmt.Println("Error in Search clEnqueueMapBuffer: ", err)
			return stop_nonce, 0
		}

		d.queue.Flush()
		if d.openCL12 {
			err := cl.WaitForEvents([]*cl.Event{preReturnEvent})
			if err != nil {
				fmt.Println("Error in Search clWaitForEvents: ", err)
				return 0, 0
			}
		}
	}

}

func (c *OpenCLMiner) Gclasszzrate() int64 {
	return int64(atomic.LoadInt32(&c.hashRate))
}
