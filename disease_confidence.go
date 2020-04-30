package main

import(
  "math"
  "math/rand"
  "encoding/json"
  "fmt"
  "flag"
  "time"
  "os"
	"io/ioutil"
  "strings"
)


type Dynamics struct{
  TimePoints []float64
  Symptomatic []float64
  Asymptomatic []float64
  NonInfected []float64
}

type Samples struct{
  DailyPositive []float64
  DailyTotal []float64
  DayTimes []float64
}

type PeakSum struct{
  RealPeaks []float64
  FoundPeaks []float64
  FoundOn []float64
  Distances []float64
  SumSqDistance float64
}

type FullResults struct{
  Peaks map[string][]PeakSum
  SumSqDistance map[string]float64
}

func fileExists(filename string) bool {
    info, err := os.Stat(filename)
    if os.IsNotExist(err) {
        return false
    }
    return !info.IsDir()
}

func max(arr []float64) float64{
  mx := arr[0]
  for i := 1; i<len(arr); i++{
    if arr[i]>mx{
      mx = arr[i]
    }
  }
  return mx
}

func Missing(slice []int, val int) bool {
    for _, item := range slice {
        if item == val {
            return false
        }
    }
    return true
}

func Smooth(x []float64,y []float64, smth float64) []float64{
  smoothed :=make([]float64, len(x))
  var smx float64
  for i:=0; i<len(x); i++{
    smx = 0
    for j:=0; j<len(x); j++{
      smx += y[j]*(1/(math.Sqrt(2*math.Pi)*smth))*math.Exp(-math.Pow(x[i]-x[j],2)/(2*smth))
    }
    smoothed[i] = smx
  }
  return smoothed
}

func Diff(x,t []float64) []float64{
  df := make([]float64,len(x)-1)
  for i:=1; i<len(x); i++{
    df[i-1] =(x[i]-x[i-1])/(t[i]-t[i-1])
  }
  return df
}

func argNeg(arr []float64) []int{
  var indx []int
  pos := true
  for i:=0; i<len(arr); i ++{
    if arr[i]<0 && pos{
      indx = append(indx,i)
      pos = false
    }
    if arr[i] >0{
      pos = true
    }
  }
  return indx
}

func getTimeIndex(alltimes []float64, t float64) int{
  indx := 0
  for i:=0; i<len(alltimes); i++{
    if t>alltimes[i]{
      indx = i
    }
  }
  return indx
}

func GenerateSampleData(dynamics Dynamics, sample_capacity []float64, sample_bias []float64, falsePos, falseNeg float64,interval float64) Samples{
  dynLen := len(dynamics.TimePoints)
  num_intervals := int(dynamics.TimePoints[dynLen-1]/interval) + 1
  PosDay := make([]float64,num_intervals)
  TotDay := make([]float64,num_intervals)
  IntervalStart := make([]float64,num_intervals)
  max_lam := max(sample_capacity)
  simTime := rand.ExpFloat64()/max_lam
  var dt float64
  var tests_interval float64
  var positive_interval float64
  for i := 0; i<num_intervals; i++{
    tests_interval = 0.0
    positive_interval = 0.0
    IntervalStart[i] = float64(i)*interval
    for simTime < float64((i+1))*interval{
      indx := getTimeIndex(dynamics.TimePoints,simTime)
      BiasN := sample_bias[indx]*dynamics.Symptomatic[indx] + dynamics.Asymptomatic[indx] + dynamics.NonInfected[indx]
      lam1 := sample_capacity[indx]*(sample_bias[indx]*dynamics.Symptomatic[indx] + dynamics.Asymptomatic[indx])/BiasN
      lam2 := sample_capacity[indx]*dynamics.NonInfected[indx]/BiasN

      u := rand.Float64()

      switch{
      case u < (falseNeg*lam1)/max_lam://false negative
        tests_interval += 1.0
      case u < (lam1/max_lam)://true positive
        tests_interval += 1.0
        positive_interval += 1.0
      case u < (lam1 + falsePos*lam2)/max_lam://false positive
        tests_interval += 1.0
        positive_interval += 1.0
      case u < sample_capacity[indx]/max_lam://true negative
        tests_interval += 1.0
      }

      dt = rand.ExpFloat64()/max_lam
      simTime += dt
    }
    PosDay[i] = positive_interval
    TotDay[i] = tests_interval
  }
  return Samples{PosDay,TotDay,IntervalStart}
}

func findPeak_smthD(samp Samples, realdyn Dynamics, smoothing float64) ([]float64,[]int,[]int){
  realinfected := make([]float64, len(realdyn.TimePoints))
  for i := 0; i < len(realdyn.TimePoints); i++{
    realinfected[i] = realdyn.Symptomatic[i] + realdyn.Asymptomatic[i]
  }
  realDiff := Diff(realinfected,realdyn.TimePoints)

  realpeakindx := argNeg(realDiff)
  realpeaktime := make([]float64,len(realpeakindx))
  for i,val := range realpeakindx{
    realpeaktime[i] = realdyn.TimePoints[val]
  }

  sampleRatio := make([]float64, len(samp.DayTimes))

  for i,val := range samp.DailyPositive{
    sampleRatio[i] = val/samp.DailyTotal[i]
  }

  var dataPeaksIndx []int
  var foundOnIndx []int
  var tempPk []int

  sampleDiff := Diff(sampleRatio,samp.DayTimes)

  for i := 19; i<len(sampleRatio); i++{
    tempPk = argNeg(Smooth(samp.DayTimes[1:i],sampleDiff[:i],smoothing))
    if len(tempPk) >0{
      dataPeaksIndx = append(dataPeaksIndx,tempPk[len(tempPk)-1]+1)
      foundOnIndx = append(foundOnIndx,i)
    }
  }

  // dataPeaks := make([]float64,len(dataPeaksIndx))
  // for i,val := range dataPeaksIndx{
  //   dataPeaks[i] = samp.DayTimes[val]
  // }
  //
  // peaksFound := make([]float64, len(foundOnIndx))
  // for i,val := range foundOnIndx{
  //   peaksFound[i] = samp.DayTimes[val]
  // }

  return realpeaktime,dataPeaksIndx,foundOnIndx
}

func ComparePeaks(realpeaks []float64,dataPeaks,PeaksFoundOn[]int, samp Samples) PeakSum{
  switch{
  case len(dataPeaks) >0:
    var found_peaks_trimmed []int
    var foundOn_day_trimmed []int
    for i,val := range dataPeaks{
      chk := Missing(found_peaks_trimmed,val)
      if chk{
        found_peaks_trimmed = append(found_peaks_trimmed,val)
        foundOn_day_trimmed = append(foundOn_day_trimmed,PeaksFoundOn[i])
      }
    }

    dataPeaksDay := make([]float64,len(found_peaks_trimmed))
    for i,val := range found_peaks_trimmed{
      dataPeaksDay[i] = samp.DayTimes[val]
    }

    peaksFoundDay := make([]float64, len(foundOn_day_trimmed))
    for i,val := range foundOn_day_trimmed{
      peaksFoundDay[i] = samp.DayTimes[val]
    }

    distance := make([]float64,len(peaksFoundDay))
    for i,val := range peaksFoundDay{
      d1 := realpeaks[0] - val//positive is early, negative is late
      for _,val2 := range realpeaks{
        d2 := val2 - val
        if d2 < d1{
          d1 = d2
        }
      }
      distance[i] = d1
    }

    smsqdist := SumSqs(distance)

    return PeakSum{realpeaks,dataPeaksDay,peaksFoundDay,distance,smsqdist}

  default:
    return PeakSum{realpeaks,[]float64{},[]float64{},[]float64{},realpeaks[0]}
  }
}

func SumSqs(arr []float64) float64{
  var smsq float64
  for _,val := range arr{
    smsq += math.Pow(val,2)
  }
  return smsq
}

func Avg(arr []float64) float64{
  var tot float64
  for _,val := range arr{
    tot += val
  }
  return tot/float64(len(arr))
}


func main(){
  var reportall bool
  flag.BoolVar(&reportall,"Verbose",false,"whether or not to print all trials in detail.")

  var allgo bool
  flag.BoolVar(&allgo,"ComputePeaks",false,"whether or not to compute peak predictions.")

  var savefl string
  flag.StringVar(&savefl,"SaveFile","json_io/out","name of save file")

  var dynamicsfl string
  flag.StringVar(&dynamicsfl,"Dynamics","",".json file with dynamics saved.")

  var testing_bias string
  flag.StringVar(&testing_bias,"TestingBias","",".json file with testing bias saved.")

  var capflnm string
  flag.StringVar(&capflnm,"TestingCapacities","",".json file with testing capacities saved.")

  var falsepositive float64
  flag.Float64Var(&falsepositive,"FalsePositive",0,"False positive test rate")

  var falsenegative float64
  flag.Float64Var(&falsenegative,"FalseNegative",0,"False negative test rate")

  var daylength float64
  flag.Float64Var(&daylength,"Interval",1,"Length of data bucket intervals")

  var smthness float64
  flag.Float64Var(&smthness,"Smoothing",5,"Guassian smoothing parameter for derivative estimation")

  var numTri int
  flag.IntVar(&numTri,"Trials",1,"Number of trials for estimation")

  flag.Parse()

  rand.Seed(time.Now().UnixNano())

  var dynamic_sim Dynamics

  switch fileExists(dynamicsfl){
    case true:
      simfl, mess1 := os.Open(dynamicsfl)
      if mess1 != nil{
        fmt.Println(mess1)
      }
      simRead, mess2 := ioutil.ReadAll(simfl)
      if mess1 != nil{
        fmt.Println(mess2)
      }
      json.Unmarshal(simRead,&dynamic_sim)
    case false:
      fmt.Println("Give Dynamics as .json file.")
  }

  var test_bias []float64

  switch fileExists(testing_bias){
    case true:
      biasfl, mess1 := os.Open(testing_bias)
      if mess1 != nil{
        fmt.Println(mess1)
      }
      biasRead, mess2 := ioutil.ReadAll(biasfl)
      if mess1 != nil{
        fmt.Println(mess2)
      }
      json.Unmarshal(biasRead,&test_bias)
    case false:
      fmt.Println("Give testing bias as .json file.")
  }

  var capacities map[string][]float64

  switch fileExists(capflnm){
    case true:
      capfl, mess1 := os.Open(capflnm)
      if mess1 != nil{
        fmt.Println(mess1)
      }
      capRead, mess2 := ioutil.ReadAll(capfl)
      if mess1 != nil{
        fmt.Println(mess2)
      }
      json.Unmarshal(capRead,&capacities)
    case false:
      fmt.Println("Give testing capacity as .json file.")
  }


  // capacities := []float64{100,500,1000,2500,5000}

  var sample_disase Samples
  // var peak_error float64

  pkRes := make(map[string][]PeakSum)
  pkErrs := make(map[string]float64)

  if allgo{
    for nm,test_caps := range capacities{
      fmt.Println("Avg Tests Per Day:",nm)
      fmt.Println("\n")
      smsqerrs := make([]float64,numTri)
      peakRes := make([]PeakSum,numTri)
      for trial := 0; trial < numTri; trial ++{
        sample_disase = GenerateSampleData(dynamic_sim,test_caps,test_bias,falsepositive,falsenegative,daylength)
        pk1,pk2,pk3 := findPeak_smthD(sample_disase, dynamic_sim, smthness)
        comp := ComparePeaks(pk1,pk2,pk3,sample_disase)
        peakRes[trial] = comp
        smsqerrs[trial] = comp.SumSqDistance
        if reportall{
          fmt.Println("Real peaks comes at days:",comp.RealPeaks)
          fmt.Println("Peaks identified as days",comp.FoundPeaks)
          fmt.Println("Those peaks found on days",comp.FoundOn)
          fmt.Println("Sum of square error:", comp.SumSqDistance)
          fmt.Println("\n")
        }
      }
      avgErr := Avg(smsqerrs)
      pkRes[nm] = peakRes
      pkErrs[nm] = avgErr
      fmt.Println("Average sum-square error:",avgErr)
      fmt.Println("\n")

      // }
    }
    Results := FullResults{pkRes,pkErrs}

    outfl,err := json.Marshal(Results)
    if err != nil{
      fmt.Println("JSON enconding error:",err)
    }

    var flnm1 string
    if strings.HasSuffix(savefl,".json"){
      flnm1 = savefl
    }else{
      flnm1 = savefl + ".json"
    }

    _ = ioutil.WriteFile(flnm1,outfl,0644)
  }else{
    Results := make(map[string][]Samples)
    for nm,test_caps := range capacities{
      these_results := make([]Samples,numTri)
      // fmt.Println("Avg Tests Per Day:", nm)
      // fmt.Println("\n")
      for trial := 0; trial < numTri; trial ++{
        sample_disase = GenerateSampleData(dynamic_sim,test_caps,test_bias,falsepositive,falsenegative,daylength)
        these_results[trial] = sample_disase
      }
      Results[nm] = these_results

      // }
    }

    outfl,err := json.Marshal(Results)
    if err != nil{
      fmt.Println("JSON enconding error:",err)
    }

    var flnm1 string

    if strings.HasSuffix(savefl,".json"){
      flnm1 = savefl
    }else{
      flnm1 = savefl + ".json"
    }

    _ = ioutil.WriteFile(flnm1,outfl,0644)
  }
}
