package main

import (
	"flag"
	"github.com/classzz/classzz/rpcclient"
	"github.com/classzz/miner-gpu/czzhash"
	"log"
	"math/big"
	"sync"
)

type Miner struct {
	Hash     string
	Target   *big.Int
	Client   *rpcclient.Client
	Cl       *czzhash.OpenCLMiner
	Nonce    chan uint64
	Stop     chan struct{}
	CancelWg sync.WaitGroup
}

// miner
func (m *Miner) mining() {

	for {
		m.Stop = make(chan struct{})
		work, err := m.Client.GetWork()
		if err != nil {
			log.Fatal("GetWork ", "err", err)
			continue
		}
		log.Println("GetWork", "Hash", work.Hash, "Target", work.Target)

		m.Hash = work.Hash
		target := big.NewInt(0).SetBytes([]byte(work.Target))
		m.Target = target

		hash := czzhash.Hash{}
		hash.SetBytes([]byte(m.Hash))

		//Hashrate
		fetchers := []func() *czzhash.Result{}

		for i := 0; i < m.Cl.GetDeviceCount(); i++ {
			index := big.NewInt(int64(i))
			fetchers = append(fetchers, func() *czzhash.Result { return m.Cl.Search(hash, m.Target.Uint64(), m.Stop, index.Int64()) })
		}

		result := make(chan *czzhash.Result, len(fetchers))
		m.CancelWg.Add(len(fetchers))
		for _, fn := range fetchers {
			fn := fn
			go func() {
				defer m.CancelWg.Done()
				result <- fn()
			}()
		}

		hashRate := uint64(0)
		Nonce := uint64(0)
		for i := 0; i < len(fetchers); i++ {
			var result_ *czzhash.Result
			if result_ = <-result; err == nil && Nonce == 0 {
				Nonce = result_.Nonce
				close(m.Stop)
			}
			hashRate = hashRate + result_.HashRate
		}

		log.Println("SubmitWork", "Nonce:", Nonce, "hashRate:", hashRate)
		if err = m.Client.SubmitWork(m.Hash, Nonce); err != nil {
			log.Fatal("SubmitWork", "err", err)
		}

	}
}

func main() {

	var HostFlag = flag.String("h", "127.0.0.1:8334", "rpcclient Host ")
	var UserFlag = flag.String("u", "", "User")
	var PassFlag = flag.String("p", "", "Pass")

	flag.Parse()

	var user string
	var pass string

	if *UserFlag != "" || *PassFlag != "" {
		user, pass = *UserFlag, *PassFlag
	}

	connCfg := &rpcclient.ConnConfig{
		Host:         *HostFlag,
		Endpoint:     "http",
		User:         user,
		Pass:         pass,
		HTTPPostMode: true, // Bitcoin core only supports HTTP POST mode
		DisableTLS:   true, // Bitcoin core does not provide TLS by default
	}

	// Notice the notification parameter is nil since notifications are
	// not supported in HTTP POST mode.
	client, err := rpcclient.New(connCfg, nil)
	if err != nil {
		log.Fatal("rpcclient", "err", err)
	}
	defer client.Shutdown()

	driverCount := czzhash.GetDeviceCount()
	cls := []int{}
	for i := 0; i < driverCount; i++ {
		cls = append(cls, i)
	}
	Cl := czzhash.NewCL(cls)
	err = czzhash.InitCL(0, Cl)

	min := &Miner{
		Cl:     Cl,
		Client: client,
		Stop:   make(chan struct{}),
	}

	min.mining()
}
