#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/wifi-module.h"
#include "ns3/internet-module.h"
#include "ns3/aodv-module.h"
#include "ns3/applications-module.h"
#include "ns3/flow-monitor-module.h"

#include <fstream>
#include <iomanip>
#include <cmath>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("ManetSimulation");

static std::string g_outDir = ".";
static uint32_t g_runId = 0;

// Simple log-distance RSSI approximation + stochastic shadowing noise.
// This is intentionally lightweight (doesn't require PHY tracing) but avoids a purely deterministic
// geometry->RSSI mapping which makes prediction unrealistically easy.
double ApproximateRssiDbm(double txPowerDbm, double distanceMeters, double pathLossExp = 3.0) {
    if (distanceMeters < 0.1) distanceMeters = 0.1;
    double loss = 10.0 * pathLossExp * std::log10(distanceMeters);
    static Ptr<NormalRandomVariable> shadowing = CreateObject<NormalRandomVariable>();
    shadowing->SetAttribute("Mean", DoubleValue(0.0));
    shadowing->SetAttribute("Variance", DoubleValue(16.0)); // sigma=4 dB (log-normal shadowing proxy)
    const double noise_db = shadowing->GetValue();
    return (txPowerDbm - loss) + noise_db;
}

void SamplePositionsAndWrite(NodeContainer nodes, double sampleTimeSec, double neighborRadius, double txPowerDbm) {
    std::ostringstream fname;
    fname << g_outDir << "/positions_run" << g_runId << ".csv";
    
    std::ofstream ofs;
    if (sampleTimeSec <= 1.0) {
        ofs.open(fname.str(), std::ofstream::out);
        ofs << "time,nodeId,x,y,neighbor_count,avg_neighbor_rssi_dbm\n";
    } else {
        ofs.open(fname.str(), std::ofstream::out | std::ofstream::app);
    }

    uint32_t n = nodes.GetN();
    std::vector<Vector> pos(n);
    for (uint32_t i = 0; i < n; ++i) {
        pos[i] = nodes.Get(i)->GetObject<MobilityModel>()->GetPosition();
    }

    for (uint32_t i = 0; i < n; ++i) {
        uint32_t neighbors = 0;
        double rssi_sum = 0.0;
        for (uint32_t j = 0; j < n; ++j) {
            if (i == j) continue;
            double dist = std::sqrt(std::pow(pos[i].x - pos[j].x, 2) + std::pow(pos[i].y - pos[j].y, 2));
            if (dist <= neighborRadius) {
                neighbors++;
                rssi_sum += ApproximateRssiDbm(txPowerDbm, dist);
            }
        }
        double avg_rssi = (neighbors > 0) ? (rssi_sum / neighbors) : -1000.0;
        ofs << std::fixed << std::setprecision(3) 
            << sampleTimeSec << "," << i << "," << pos[i].x << "," << pos[i].y << "," 
            << neighbors << "," << avg_rssi << "\n";
    }
    ofs.close();
}

int main(int argc, char *argv[]) {
    uint32_t numNodes = 30;
    double simTimeSeconds = 60.0;
    uint32_t rngRun = 1;
    std::string outDir = ".";
    double commRadiusMeters = 150.0;

    CommandLine cmd;
    cmd.AddValue("numNodes", "Number of nodes", numNodes);
    cmd.AddValue("runId", "Run identifier", g_runId);
    cmd.AddValue("RngRun", "RNG run seed", rngRun);
    cmd.AddValue("outDir", "Output directory", outDir);
    cmd.AddValue("commRadius", "Communication/neighborhood radius (m) used for sampling", commRadiusMeters);
    cmd.Parse(argc, argv);

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
    positionAlloc->SetAttribute("X", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=500.0]"));
    positionAlloc->SetAttribute("Y", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=500.0]"));

    // Ensure nodes have a valid random initial position at t=0 (avoid all nodes starting at (0,0)).
    mobility.SetPositionAllocator(positionAlloc);
    mobility.SetMobilityModel(
        "ns3::RandomWaypointMobilityModel",
        "Speed", StringValue("ns3::UniformRandomVariable[Min=5.0|Max=20.0]"),
        "Pause", StringValue("ns3::ConstantRandomVariable[Constant=1.0]"),
        "PositionAllocator", PointerValue(positionAlloc)
    );
    mobility.Install(nodes);

    AodvHelper aodv;
    InternetStackHelper internet;
    internet.SetRoutingHelper(aodv);
    internet.Install(nodes);
    Ipv4AddressHelper ipv4;
    ipv4.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer interfaces = ipv4.Assign(devices);

    // Setup basic UDP traffic flows
    uint16_t port = 9;
    for (uint32_t i = 0; i < numNodes / 2; ++i) {
        UdpEchoServerHelper echoServer(port);
        ApplicationContainer serverApps = echoServer.Install(nodes.Get(numNodes - 1 - i));
        serverApps.Start(Seconds(1.0));
        serverApps.Stop(Seconds(simTimeSeconds));

        UdpEchoClientHelper echoClient(interfaces.GetAddress(numNodes - 1 - i), port);
        echoClient.SetAttribute("MaxPackets", UintegerValue(100));
        echoClient.SetAttribute("Interval", TimeValue(Seconds(1.0)));
        echoClient.SetAttribute("PacketSize", UintegerValue(1024));
        ApplicationContainer clientApps = echoClient.Install(nodes.Get(i));
        clientApps.Start(Seconds(2.0));
        clientApps.Stop(Seconds(simTimeSeconds));
    }

    FlowMonitorHelper flowmon;
    Ptr<FlowMonitor> monitor = flowmon.InstallAll();

    // Schedule position sampling
    for (double t = 1.0; t <= simTimeSeconds; t += 1.0) {
        Simulator::Schedule(Seconds(t), &SamplePositionsAndWrite, nodes, t, commRadiusMeters, 16.0);
    }

    Simulator::Stop(Seconds(simTimeSeconds));
    Simulator::Run();

    std::ostringstream xmlname;
    xmlname << outDir << "/manet_flowmon_run" << g_runId << ".xml";
    monitor->SerializeToXmlFile(xmlname.str(), true, true);

    Simulator::Destroy();
    return 0;
}