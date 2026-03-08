#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/internet-module.h"
#include "ns3/aodv-module.h"
#include "ns3/wifi-module.h"
#include "ns3/applications-module.h"
#include "ns3/netanim-module.h"
#include <fstream>
#include <cmath>
#include <string>

using namespace ns3;

// Global CSV writer and simulation parameters for dataset generation.
// The file dataset/link_dataset.csv will contain:
// nodeA,nodeB,distance,time,link_status
// where link_status = 1 if distance <= LINK_RANGE_METERS, else 0.
std::ofstream dataset;

const uint32_t NUM_NODES = 50;
const double SIM_TIME = 100.0;
const double LINK_RANGE_METERS = 100.0;
const double LOG_INTERVAL_SECONDS = 1.0;
const std::string DATASET_PATH = "dataset/link_dataset.csv";

void LogNodeDistances(NodeContainer nodes)
{
    for (uint32_t i = 0; i < nodes.GetN(); i++)
    {
        Ptr<MobilityModel> mob1 = nodes.Get(i)->GetObject<MobilityModel>();

        for (uint32_t j = i + 1; j < nodes.GetN(); j++)
        {
            Ptr<MobilityModel> mob2 = nodes.Get(j)->GetObject<MobilityModel>();

            double distance = mob1->GetDistanceFrom(mob2);

            int link_status = (distance <= LINK_RANGE_METERS) ? 1 : 0;

            dataset << i << "," << j << "," << distance << ","
                    << Simulator::Now().GetSeconds() << ","
                    << link_status << std::endl;
        }
    }

    Simulator::Schedule(Seconds(LOG_INTERVAL_SECONDS), &LogNodeDistances, nodes);
}

int main(int argc, char *argv[])
{
    // Open the CSV dataset file used later by the Python ML pipeline.
    // Make sure the "dataset" directory exists before running the simulation.
    dataset.open(DATASET_PATH.c_str());
    dataset << "nodeA,nodeB,distance,time,link_status" << std::endl;

    NodeContainer nodes;
    nodes.Create(NUM_NODES);

    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211b);

    YansWifiPhyHelper phy;
    YansWifiChannelHelper channel = YansWifiChannelHelper::Default();
    phy.SetChannel(channel.Create());

    WifiMacHelper mac;
    mac.SetType("ns3::AdhocWifiMac");

    NetDeviceContainer devices = wifi.Install(phy, mac, nodes);

    // ---------- MOBILITY ----------
    MobilityHelper mobility;

    mobility.SetPositionAllocator(
        "ns3::GridPositionAllocator",
        "MinX", DoubleValue(0.0),
        "MinY", DoubleValue(0.0),
        "DeltaX", DoubleValue(20.0),
        "DeltaY", DoubleValue(20.0),
        "GridWidth", UintegerValue(10),
        "LayoutType", StringValue("RowFirst")
    );

    mobility.SetMobilityModel("ns3::RandomWalk2dMobilityModel",
        "Bounds", RectangleValue(Rectangle(0,1000,0,1000)));

    mobility.Install(nodes);
    // --------------------------------

    AodvHelper aodv;

    InternetStackHelper internet;
    internet.SetRoutingHelper(aodv);
    internet.Install(nodes);

    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");

    Ipv4InterfaceContainer interfaces = address.Assign(devices);

    UdpEchoServerHelper server(9);
    ApplicationContainer serverApp = server.Install(nodes.Get(0));
    serverApp.Start(Seconds(1.0));
    serverApp.Stop(Seconds(SIM_TIME));

    UdpEchoClientHelper client(interfaces.GetAddress(0), 9);

    client.SetAttribute("MaxPackets", UintegerValue(100));
    client.SetAttribute("Interval", TimeValue(Seconds(1.0)));
    client.SetAttribute("PacketSize", UintegerValue(512));

    ApplicationContainer clientApp = client.Install(nodes.Get(10));
    clientApp.Start(Seconds(2.0));
    clientApp.Stop(Seconds(SIM_TIME));

    AnimationInterface anim("manet-animation.xml");
    Simulator::Stop(Seconds(SIM_TIME));

    Simulator::Schedule(Seconds(LOG_INTERVAL_SECONDS), &LogNodeDistances, nodes);
    Simulator::Run();
    Simulator::Destroy();

    dataset.close();
    return 0;
}