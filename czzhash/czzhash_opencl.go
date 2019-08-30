package czzhash

import (
	crand "crypto/rand"
	"fmt"
	"github.com/Gustav-Simonsson/go-opencl/cl"
	"log"
	"math"
	"math/big"
	"math/rand"
	"sync"
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
	result        Hash
}

type OpenCLMiner struct {
	mu sync.Mutex

	czzhash *CzzHash // classzz full Bin & cache in host mem

	deviceIds []int
	devices   []*OpenCLDevice

	binSize uint64
}

type Result struct {
	HashRate uint64
	Nonce    uint64
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
		fmt.Printf("Device (%s): %s", devices[id].Type(), devices[id].Name())
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
		return fmt.Errorf("opencl version not supported %s", device.Version())
	}
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

	log.Println("Initialising device", deviceId, device.Name())
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

func (c *OpenCLMiner) Search(hash [32]byte, target uint64, stop <-chan struct{}, index int64) *Result {

	headerHash := hash
	log.Println("Search", "index:", index, "hash:", hash, "target:", target)

	d := c.devices[index]
	headerBuf, err := d.ctx.CreateEmptyBuffer(cl.MemReadOnly, 32)
	_, err = d.queue.EnqueueWriteBuffer(headerBuf, true, 0, 32, unsafe.Pointer(&headerHash), nil)
	if err != nil {
		log.Fatal("Error in Search clEnqueueWriterBuffer", "err", err)
		return nil
	}

	defer d.queue.Flush()

	// we grab a single random nonce and sets this as argument to the kernel search function
	// the device will then add each local threads gid to the nonce, creating a unique nonce
	// for each device computing unit executing in parallel
	seed, _ := crand.Int(crand.Reader, big.NewInt(math.MaxInt64))
	InitNonce := rand.New(rand.NewSource(seed.Int64())).Uint64()
	Nonce := InitNonce

	for {
		select {
		case <-stop:
			su := &Result{
				HashRate: Nonce - InitNonce,
			}
			return su
		default:
			err = d.searchKernel.SetArg(1, headerBuf)
			if err != nil {
				log.Fatal("Error in Search clSetKernelArg ", "err", err)
				return nil
			}

			err = d.searchKernel.SetArg(2, d.binBuf)
			if err != nil {
				log.Fatal("Error in Search clSetKernelArg ", "err", err)
				return nil
			}

			err = d.searchKernel.SetArg(4, target)
			if err != nil {
				log.Fatal("Error in Search clSetKernelArg ", "err", err)
				return nil
			}

			isolate := uint32(0xFFFFFFFF)
			err = d.searchKernel.SetArg(5, isolate)
			if err != nil {
				log.Fatal("Error in Search clSetKernelArg ", "err", err)
				return nil
			}

			err = d.searchKernel.SetArg(0, d.searchBuffer)
			if err != nil {
				log.Fatal("Error in Search clSetKernelArg ", "err", err)
				return nil
			}
			err = d.searchKernel.SetArg(3, Nonce)
			if err != nil {
				log.Fatal("Error in Search clSetKernelArg ", "err", err)
				return nil
			}

			// execute kernel
			_, err = d.queue.EnqueueNDRangeKernel(
				d.searchKernel,
				nil,
				[]int{1},
				[]int{1},
				nil)

			if err != nil {
				log.Fatal("Error in Search clEnqueueNDRangeKernel ", "err", err)
				return nil
			}

			var result [32]byte
			_, err = d.queue.EnqueueReadBuffer(d.searchBuffer, true, 0, 32, unsafe.Pointer(&result), nil)
			if err != nil {
				log.Fatal("Error in Search clEnqueueMapBuffer ", "err", err)
				return nil
			}
			//log.Println("index",index,"result ",result)
			if new(big.Int).SetBytes(result[:]).Cmp(big.NewInt(0).SetUint64(target)) <= 0 {
				su := &Result{
					HashRate: Nonce - InitNonce,
					Nonce:    Nonce,
				}
				return su
			}
		}
		Nonce++
	}
}

func (c *OpenCLMiner) GetDeviceCount() int {
	return len(c.devices)
}

func GetDeviceCount() int {

	platforms, err := cl.GetPlatforms()
	if err != nil {
		return 0
	}
	count := 0
	for _, p := range platforms {
		ds, err := cl.GetDevices(p, cl.DeviceTypeGPU)
		if err != nil {
			return 0
		}
		for _, d := range ds {
			d.Name()
			count++
		}
	}
	return count
}
