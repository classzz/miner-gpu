package main

import (
	"flag"
	"fmt"
	"github.com/classzz/classzz/rpcclient"
	"github.com/classzz/miner-gpu/czzhash"
	"log"
	"math/big"
)

type Miner struct {
	Hash   string
	Target *big.Int
	Client *rpcclient.Client
	Cl     *czzhash.OpenCLMiner
}

// miner
func (m *Miner) mining() {

	for {
		work, err := m.Client.GetWork()
		if err != nil {
			fmt.Println("GetWork err:", err)
			continue
		}

		m.Hash = work.Hash
		target := big.NewInt(0).SetBytes([]byte(work.Target))
		m.Target = target

		hash := czzhash.Hash{}
		hash.SetBytes([]byte(m.Hash))

		//Hashrate
		Nonce, _ := m.Cl.Search(hash, m.Target.Uint64(), nil, 0)

		err = m.Client.SubmitWork(m.Hash, Nonce)
		fmt.Println("err", err)
	}
}

func main() {

	var HostFlag = flag.String("h", "127.0.0.1:8334", "rpcclient Host ")
	var UserFlag = flag.String("u", "", "User")
	var PassFlag = flag.String("p", "", "Pass")

	flag.Parse()
	//fmt.Println(*HostFlag, *UserFlag, *PassFlag)

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
		log.Fatal(err)
	}
	defer client.Shutdown()

	Cl := czzhash.NewCL([]int{1})
	err = czzhash.InitCL(0, Cl)

	min := &Miner{
		Cl:     Cl,
		Client: client,
	}

	min.mining()
}
