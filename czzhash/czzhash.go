package czzhash

import (
	"fmt"
	"io/ioutil"
	"os"
	"sync"
	"sync/atomic"
)

const (
	TBLSize    = 33554432
	HashLength = 32
)

type Csatable [TBLSize]byte
type Hash [HashLength]byte

func (c Csatable) Bytes() []byte { return c[:TBLSize] }

// Sets the hash to the value of b. If b is larger than len(h) it will panic
func (c *Csatable) SetBytes(b []byte) {
	if len(b) > len(c.Bytes()) {
		b = b[len(b)-TBLSize:]
	}
	copy(c[TBLSize-len(b):], b)
}

// SetBytes sets the hash to the value of b.
// If b is larger than len(h), b will be cropped from the left.
func (h *Hash) SetBytes(b []byte) {
	if len(b) > len(h) {
		b = b[len(b)-HashLength:]
	}
	copy(h[HashLength-len(b):], b)
}

// Full implements the Search half of the proof of work.
type Full struct {
	hashRate int32
	mu       sync.Mutex // protects bin
	Csatable *Csatable  // current full Bin
}

func (pow *Full) GetBin(blockNum uint64) *Csatable {

	if pow.Csatable != nil {
		return pow.Csatable
	}
	file, err := os.Open("csatable.bin")
	if err != nil {
		fmt.Println(err)
	}
	defer file.Close()
	stats, _ := file.Stat()
	var size = uint64(stats.Size())
	if size != TBLSize {
		fmt.Printf("file size is %d\n", size)
		return &Csatable{}
	}
	csa, err := ioutil.ReadAll(file)
	cast := &Csatable{}
	cast.SetBytes(csa)

	return cast
}

func (pow *Full) GetHashrate() int64 {
	return int64(atomic.LoadInt32(&pow.hashRate))
}

type CzzHash struct {
	*Full
}

// New creates an instance of the proof of work.
func New() *CzzHash {
	return &CzzHash{&Full{}}
}
