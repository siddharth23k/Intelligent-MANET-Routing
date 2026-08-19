#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/wifi-module.h"
#include "ns3/internet-module.h"
#include "ns3/aodv-module.h"
#include "ns3/applications-module.h"
#include "ns3/flow-monitor-module.h"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <map>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("PaperFrlfpSimulation");

static std::string g_outDir = ".";
static uint32_t g_runId = 0;

static double
ApproximateRssiDbm(double txPowerDbm, double distanceMeters, double pathLossExp = 3.0)
{
    if (distanceMeters < 0.1)
    {
        distanceMeters = 0.1;
    }
    double loss = 10.0 * pathLossExp * std::log10(distanceMeters);
    static Ptr<NormalRandomVariable> shadowing = CreateObject<NormalRandomVariable>();
    shadowing->SetAttribute("Mean", DoubleValue(0.0));
    shadowing->SetAttribute("Variance", DoubleValue(16.0));
    const double noiseDb = shadowing->GetValue();
    return (txPowerDbm - loss) + noiseDb;
}


// Periodic FlowMonitor checkpointing.
//
// FlowMonitor's XML serialisation is an end of run aggregate. Using those totals
// as a feature at time t means the value depends on packets that had not been
// sent yet, which is lookahead. Writing per interval deltas here makes the
// traffic features causal, so the Python pipeline can build a running delivery
// ratio from data available at or before t.
static std::map<uint32_t, uint64_t> g_lastTx, g_lastRx, g_lastLost;
static std::map<uint32_t, double> g_lastDelay;
static bool g_flowStatsHeaderWritten = false;

static void
SampleFlowStatsAndWrite(Ptr<FlowMonitor> monitor,
                        Ptr<Ipv4FlowClassifier> classifier,
                        double sampleTimeSec,
                        uint32_t numNodes)
{
    std::ostringstream fname;
    fname << g_outDir << "/flowstats_run" << g_runId << ".csv";

    std::ofstream ofs;
    if (!g_flowStatsHeaderWritten)
    {
        ofs.open(fname.str(), std::ofstream::out);
        ofs << "time,nodeId,tx_packets,rx_packets,lost_packets,delay_sum\n";
        g_flowStatsHeaderWritten = true;
    }
    else
    {
        ofs.open(fname.str(), std::ofstream::out | std::ofstream::app);
    }

    monitor->CheckForLostPackets();
    const FlowMonitor::FlowStatsContainer stats = monitor->GetFlowStats();

    std::vector<uint64_t> tx(numNodes, 0), rx(numNodes, 0), lost(numNodes, 0);
    std::vector<double> delay(numNodes, 0.0);

    for (auto it = stats.begin(); it != stats.end(); ++it)
    {
        const uint32_t flowId = it->first;
        Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(flowId);

        // 10.1.1.k maps back to node k-1, matching the addressing set up below.
        const uint32_t srcNode = (t.sourceAddress.Get() & 0xff) - 1;
        const uint32_t dstNode = (t.destinationAddress.Get() & 0xff) - 1;
        if (srcNode >= numNodes || dstNode >= numNodes)
        {
            continue;
        }

        const uint64_t dTx = it->second.txPackets - g_lastTx[flowId];
        const uint64_t dRx = it->second.rxPackets - g_lastRx[flowId];
        const uint64_t dLost = it->second.lostPackets - g_lastLost[flowId];
        const double dDelay = it->second.delaySum.GetDouble() - g_lastDelay[flowId];

        g_lastTx[flowId] = it->second.txPackets;
        g_lastRx[flowId] = it->second.rxPackets;
        g_lastLost[flowId] = it->second.lostPackets;
        g_lastDelay[flowId] = it->second.delaySum.GetDouble();

        tx[srcNode] += dTx;
        lost[srcNode] += dLost;
        delay[srcNode] += dDelay;
        rx[dstNode] += dRx;
    }

    for (uint32_t i = 0; i < numNodes; ++i)
    {
        ofs << std::fixed << std::setprecision(3) << sampleTimeSec << "," << i << ","
            << tx[i] << "," << rx[i] << "," << lost[i] << "," << delay[i] << "\n";
    }
    ofs.close();
}

static void
SamplePositionsAndWrite(NodeContainer nodes, double sampleTimeSec, double neighborRadius, double txPowerDbm)
{
    std::ostringstream fname;
    fname << g_outDir << "/positions_run" << g_runId << ".csv";

    std::ofstream ofs;
    if (sampleTimeSec <= 1.0)
    {
        ofs.open(fname.str(), std::ofstream::out);
        ofs << "time,nodeId,x,y,neighbor_count,avg_neighbor_rssi_dbm\n";
    }
    else
    {
        ofs.open(fname.str(), std::ofstream::out | std::ofstream::app);
    }

    uint32_t n = nodes.GetN();
    std::vector<Vector> pos(n);
    for (uint32_t i = 0; i < n; ++i)
    {
        pos[i] = nodes.Get(i)->GetObject<MobilityModel>()->GetPosition();
    }

    for (uint32_t i = 0; i < n; ++i)
    {
        uint32_t neighbors = 0;
        double rssiSum = 0.0;
        for (uint32_t j = 0; j < n; ++j)
        {
            if (i == j)
            {
                continue;
            }
            double dist = std::sqrt(std::pow(pos[i].x - pos[j].x, 2) + std::pow(pos[i].y - pos[j].y, 2));
            if (dist <= neighborRadius)
            {
                neighbors++;
                rssiSum += ApproximateRssiDbm(txPowerDbm, dist);
            }
        }
        double avgRssi = (neighbors > 0) ? (rssiSum / neighbors) : -1000.0;
        ofs << std::fixed << std::setprecision(3) << sampleTimeSec << "," << i << "," << pos[i].x << ","
            << pos[i].y << "," << neighbors << "," << avgRssi << "\n";
    }
    ofs.close();
}

int
main(int argc, char* argv[])
{
    // Paper-like defaults (table/scenario aligned and tunable).
    uint32_t numNodes = 100;
    double simTimeSeconds = 300.0;
    uint32_t rngRun = 1;
    std::string outDir = ".";
    double commRadiusMeters = 150.0;
    double areaSizeMeters = 1000.0;
    double speedMinMps = 0.0;
    double speedMaxMps = 60.0;
    double pauseSeconds = 2.0;
    uint32_t packetSizeBytes = 512;
    uint32_t maxPacketsPerFlow = 300;
    double packetIntervalSeconds = 1.0;
    uint32_t cbrConnections = 10;
    double txPowerDbm = 16.0;
    bool logFlowStats = true;

    CommandLine cmd;
    cmd.AddValue("numNodes", "Number of nodes", numNodes);
    cmd.AddValue("simTimeSeconds", "Simulation time in seconds", simTimeSeconds);
    cmd.AddValue("runId", "Run identifier", g_runId);
    cmd.AddValue("RngRun", "RNG run seed", rngRun);
    cmd.AddValue("outDir", "Output directory", outDir);
    cmd.AddValue("commRadius", "Communication/neighborhood radius (m) used for sampling", commRadiusMeters);
    cmd.AddValue("area", "Square area side length in meters", areaSizeMeters);
    cmd.AddValue("speedMin", "Minimum node speed (m/s)", speedMinMps);
    cmd.AddValue("speedMax", "Maximum node speed (m/s)", speedMaxMps);
    cmd.AddValue("pause", "Mobility pause time (s)", pauseSeconds);
    cmd.AddValue("packetSize", "UDP packet size (bytes)", packetSizeBytes);
    cmd.AddValue("maxPackets", "Max packets per source flow", maxPacketsPerFlow);
    cmd.AddValue("interval", "Inter-packet interval per flow (s)", packetIntervalSeconds);
    cmd.AddValue("cbrConnections", "Number of CBR source-destination pairs", cbrConnections);
    cmd.AddValue("txPowerDbm", "Transmit power used in RSSI approximation", txPowerDbm);
    cmd.AddValue("logFlowStats",
                 "Write per second FlowMonitor deltas so traffic features are causal",
                 logFlowStats);
    cmd.Parse(argc, argv);

    if (numNodes < 2)
    {
        numNodes = 2;
    }
    cbrConnections = std::min(cbrConnections, numNodes / 2);
    if (speedMaxMps < speedMinMps)
    {
        std::swap(speedMaxMps, speedMinMps);
    }

    g_outDir = outDir;
    RngSeedManager::SetSeed(1);
    RngSeedManager::SetRun(rngRun);

    NodeContainer nodes;
    nodes.Create(numNodes);

    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211b);
    YansWifiPhyHelper phy;
    YansWifiChannelHelper channel = YansWifiChannelHelper::Default();
    phy.SetChannel(channel.Create());
    WifiMacHelper mac;
    mac.SetType("ns3::AdhocWifiMac");
    NetDeviceContainer devices = wifi.Install(phy, mac, nodes);

    MobilityHelper mobility;
    Ptr<RandomRectanglePositionAllocator> positionAlloc = CreateObject<RandomRectanglePositionAllocator>();
    {
        std::ostringstream xSpec;
        std::ostringstream ySpec;
        xSpec << "ns3::UniformRandomVariable[Min=0.0|Max=" << areaSizeMeters << "]";
        ySpec << "ns3::UniformRandomVariable[Min=0.0|Max=" << areaSizeMeters << "]";
        positionAlloc->SetAttribute("X", StringValue(xSpec.str()));
        positionAlloc->SetAttribute("Y", StringValue(ySpec.str()));
    }
    mobility.SetPositionAllocator(positionAlloc);
    {
        std::ostringstream speedSpec;
        std::ostringstream pauseSpec;
        speedSpec << "ns3::UniformRandomVariable[Min=" << speedMinMps << "|Max=" << speedMaxMps << "]";
        pauseSpec << "ns3::ConstantRandomVariable[Constant=" << pauseSeconds << "]";
        mobility.SetMobilityModel("ns3::RandomWaypointMobilityModel",
                                  "Speed",
                                  StringValue(speedSpec.str()),
                                  "Pause",
                                  StringValue(pauseSpec.str()),
                                  "PositionAllocator",
                                  PointerValue(positionAlloc));
    }
    mobility.Install(nodes);

    AodvHelper aodv;
    InternetStackHelper internet;
    internet.SetRoutingHelper(aodv);
    internet.Install(nodes);
    Ipv4AddressHelper ipv4;
    ipv4.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer interfaces = ipv4.Assign(devices);

    uint16_t port = 9;
    for (uint32_t i = 0; i < cbrConnections; ++i)
    {
        uint32_t src = i;
        uint32_t dst = numNodes - 1 - i;
        if (src == dst)
        {
            continue;
        }

        UdpEchoServerHelper server(port);
        ApplicationContainer serverApps = server.Install(nodes.Get(dst));
        serverApps.Start(Seconds(1.0));
        serverApps.Stop(Seconds(simTimeSeconds));

        UdpEchoClientHelper client(interfaces.GetAddress(dst), port);
        client.SetAttribute("MaxPackets", UintegerValue(maxPacketsPerFlow));
        client.SetAttribute("Interval", TimeValue(Seconds(packetIntervalSeconds)));
        client.SetAttribute("PacketSize", UintegerValue(packetSizeBytes));
        ApplicationContainer clientApps = client.Install(nodes.Get(src));
        clientApps.Start(Seconds(2.0));
        clientApps.Stop(Seconds(simTimeSeconds));
    }

    FlowMonitorHelper flowmon;
    Ptr<FlowMonitor> monitor = flowmon.InstallAll();
    Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(flowmon.GetClassifier());

    for (double t = 1.0; t <= simTimeSeconds; t += 1.0)
    {
        Simulator::Schedule(Seconds(t), &SamplePositionsAndWrite, nodes, t, commRadiusMeters, txPowerDbm);
        if (logFlowStats)
        {
            Simulator::Schedule(Seconds(t),
                                &SampleFlowStatsAndWrite,
                                monitor,
                                classifier,
                                t,
                                numNodes);
        }
    }

    Simulator::Stop(Seconds(simTimeSeconds));
    Simulator::Run();

    std::ostringstream xmlName;
    xmlName << outDir << "/manet_flowmon_run" << g_runId << ".xml";
    monitor->SerializeToXmlFile(xmlName.str(), true, true);

    Simulator::Destroy();
    return 0;
}
