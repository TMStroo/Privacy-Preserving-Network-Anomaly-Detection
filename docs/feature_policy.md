# Feature Policy Documentation


Privacy-Preserving Network Anomaly Detection - Feature Policy
==============================================================

CORE PRINCIPLE: Only traffic metadata features are used. No packet payload
inspection, no deep packet inspection, no content analysis.

ALLOWED FEATURE CATEGORIES:
---------------------------
1. Flow Duration & Timing
   - dur: Flow duration (seconds)
   - sinpkt: Source inter-packet arrival time (mean)
   - dinpkt: Destination inter-packet arrival time (mean)
   - sjit: Source jitter (inter-packet arrival time variance)
   - djit: Destination jitter
   - tcprtt: TCP round-trip time (from handshake)
   - synack: Time between SYN and SYN-ACK
   - ackdat: Time between SYN-ACK and ACK

2. Packet & Byte Counts
   - spkts: Source-to-destination packet count
   - dpkts: Destination-to-source packet count
   - sbytes: Source-to-destination byte count
   - dbytes: Destination-to-source byte count

3. Rate & Throughput
   - rate: Packets per second
   - sload: Source-to-destination bits per second
   - dload: Destination-to-source bits per second

4. Packet Size Statistics
   - smean: Mean packet size (source to destination)
   - dmean: Mean packet size (destination to source)

5. TCP Connection State
   - sttl: Source TTL
   - dttl: Destination TTL
   - swin: Source TCP window size
   - dwin: Destination TCP window size
   - stcpb: Source TCP sequence number base
   - dtcpb: Destination TCP sequence number base

6. Loss & Retransmission
   - sloss: Source packet loss/retransmission count
   - dloss: Destination packet loss/retransmission count

7. Connection Tracking (behavioral aggregates)
   - ct_srv_src: Connections to same service from same source
   - ct_state_ttl: Connections with same state and TTL
   - ct_dst_ltm: Connections to same destination (last time window)
   - ct_src_dport_ltm: Connections from same source to same dest port
   - ct_dst_sport_ltm: Connections to same destination from same src port
   - ct_dst_src_ltm: Connections between same dest and src
   - ct_src_ltm: Connections from same source (last time window)
   - ct_srv_dst: Connections to same service from same destination
   - is_sm_ips_ports: Same source/dest IP and port (loopback)

8. Protocol & Service Identification
   - proto: Transport protocol (tcp, udp, icmp, etc.)
   - service: Application service label (http, ftp, dns, smtp, etc.)
   - state: Connection state (FIN, CON, REQ, RST, etc.)

EXPLICITLY EXCLUDED FEATURES:
-----------------------------
- srcip, dstip: IP addresses (identity-bearing, not generalizable)
- sport, dsport: Source/destination port numbers (identity-bearing)
- srcpt, dstpt: Same as above (different naming in some versions)
- Ltime, Stime: Absolute timestamps (enable tracking across flows)
- attack_cat, label: Labels - the prediction target, never a feature
- id: Row index from the dataset release - position, not traffic behavior
- Any field containing payload text, strings, or content inspection

EXCLUDED AS PAYLOAD-DERIVED (present in UNSW-NB15, rejected here):
-------------------------------------------------------------------
- trans_depth: application-layer transaction depth (needs HTTP parsing)
- response_body_len: HTTP Content-Length (application header, inside TLS)
- is_ftp_login: FTP login flag (FTP control-channel commands)
- ct_ftp_cmd: FTP command counter (FTP control-channel commands)
- ct_flw_http_mthd: HTTP method counter (HTTP request line)

These five fields exist only because UNSW-NB15's capture tooling read
application-layer content. With encrypted transports that content is not
visible, so keeping them would make the "payload-free" claim false.

NOTES ON FIELDS THAT ARE ALLOWED BUT DERIVE FROM HEADERS:
- ct_src_dport_ltm, ct_dst_sport_ltm, is_sm_ips_ports are aggregate counts
  keyed on addresses/ports. The raw addresses and ports never enter the model;
  only the counts do, and a flow exporter produces them from headers alone.
- service is assigned by capture tooling from protocol/port mapping, which a
  flow exporter also produces without reading payload.

RATIONALE FOR EACH FEATURE CATEGORY:
------------------------------------
All allowed features are observable from:
- Flow records (NetFlow, IPFIX, sFlow)
- Packet headers only (no payload)
- Connection state tracking
- Behavioral aggregates over time windows

These are exactly the features available to a network monitor that only
sees packet headers and flow records. Fields that additionally require a
visible TCP handshake (tcprtt, synack, ackdat) are noted as such in the
per-feature documentation; on encrypted transports the handshake itself is
still visible, only the payload is not.


## Feature Set Definitions


### FULL_METADATA (37 features)

| Feature | Description | Category |
|---------|-------------|----------|
| dur | Flow duration in seconds. Observable from flow start/end timestamps in packet headers. | Timing |
| spkts | Source-to-destination packet count. Countable from packet headers. | Volume/Rate |
| dpkts | Destination-to-source packet count. Countable from packet headers. | Volume/Rate |
| sbytes | Source-to-destination byte count. Sum of IP packet lengths from headers. | Volume/Rate |
| dbytes | Destination-to-source byte count. Sum of IP packet lengths from headers. | Volume/Rate |
| rate | Packets per second (spkts+dpkts)/dur. Derived from counts and duration. | Volume/Rate |
| sttl | Source TTL value from IP header. Directly observable. | TCP State |
| dttl | Destination TTL value from IP header. Directly observable. | TCP State |
| sload | Source-to-destination bit rate. Derived from sbytes and dur. | Volume/Rate |
| dload | Destination-to-source bit rate. Derived from dbytes and dur. | Volume/Rate |
| sloss | Source packet loss/retransmissions. Inferred from TCP sequence numbers. | TCP State |
| dloss | Destination packet loss/retransmissions. Inferred from TCP sequence numbers. | TCP State |
| sinpkt | Source inter-packet arrival time (mean). From packet timestamps. | Timing |
| dinpkt | Destination inter-packet arrival time (mean). From packet timestamps. | Timing |
| sjit | Source jitter (variance in inter-arrival). From packet timestamps. | Timing |
| djit | Destination jitter. From packet timestamps. | Timing |
| swin | Source TCP window size. From TCP header. | TCP State |
| dwin | Destination TCP window size. From TCP header. | TCP State |
| stcpb | Source TCP sequence number base. From TCP header. | TCP State |
| dtcpb | Destination TCP sequence number base. From TCP header. | TCP State |
| tcprtt | TCP round-trip time from handshake. From SYN/SYN-ACK timestamps. | Timing |
| synack | Time between SYN and SYN-ACK. From packet timestamps. | Timing |
| ackdat | Time between SYN-ACK and ACK. From packet timestamps. | Timing |
| smean | Mean packet size (source->dest). sbytes/spkts. | Packet Size |
| dmean | Mean packet size (dest->source). dbytes/dpkts. | Packet Size |
| ct_srv_src | Connection count: same service, same source (time window). Behavioral aggregate. | Behavioral Aggregate |
| ct_state_ttl | Connection count: same state, same TTL. Behavioral aggregate. | Behavioral Aggregate |
| ct_dst_ltm | Connection count: same destination (last time window). Behavioral aggregate. | Behavioral Aggregate |
| ct_src_dport_ltm | Connection count: same source, same dest port (last time window). Aggregate count only; ports never enter the model. | Behavioral Aggregate |
| ct_dst_sport_ltm | Connection count: same dest, same src port (last time window). Aggregate count only; ports never enter the model. | Behavioral Aggregate |
| ct_dst_src_ltm | Connection count: same dest and src pair (last time window). Keyed on header addresses; only the count is used. | Behavioral Aggregate |
| ct_src_ltm | Connection count: same source (last time window). Behavioral aggregate. | Behavioral Aggregate |
| ct_srv_dst | Connection count: same service, same destination. Behavioral aggregate. | Behavioral Aggregate |
| is_sm_ips_ports | Same source/dest IP and port indicator. Aggregate derived from headers; raw addresses never enter the model. | Behavioral Aggregate |
| proto | Transport protocol (tcp/udp/icmp). From IP header protocol field. | Categorical |
| service | Application service label. Assigned by capture tooling from protocol/port mapping, obtainable from headers without payload. | Categorical |
| state | Connection state (FIN/CON/REQ/RST/etc.). From TCP flags in headers. | Categorical |

### RESTRICTED_METADATA (17 features)

| Feature | Description | Category |
|---------|-------------|----------|
| dur | Flow duration in seconds. Observable from flow start/end timestamps in packet headers. | Timing |
| spkts | Source-to-destination packet count. Countable from packet headers. | Volume/Rate |
| dpkts | Destination-to-source packet count. Countable from packet headers. | Volume/Rate |
| sbytes | Source-to-destination byte count. Sum of IP packet lengths from headers. | Volume/Rate |
| dbytes | Destination-to-source byte count. Sum of IP packet lengths from headers. | Volume/Rate |
| rate | Packets per second (spkts+dpkts)/dur. Derived from counts and duration. | Volume/Rate |
| sttl | Source TTL value from IP header. Directly observable. | TCP State |
| dttl | Destination TTL value from IP header. Directly observable. | TCP State |
| sload | Source-to-destination bit rate. Derived from sbytes and dur. | Volume/Rate |
| dload | Destination-to-source bit rate. Derived from dbytes and dur. | Volume/Rate |
| sinpkt | Source inter-packet arrival time (mean). From packet timestamps. | Timing |
| dinpkt | Destination inter-packet arrival time (mean). From packet timestamps. | Timing |
| sjit | Source jitter (variance in inter-arrival). From packet timestamps. | Timing |
| djit | Destination jitter. From packet timestamps. | Timing |
| tcprtt | TCP round-trip time from handshake. From SYN/SYN-ACK timestamps. | Timing |
| synack | Time between SYN and SYN-ACK. From packet timestamps. | Timing |
| ackdat | Time between SYN-ACK and ACK. From packet timestamps. | Timing |

## Excluded Features

| Feature | Reason for Exclusion |
|---------|---------------------|
| srcip | IP addresses - identity-bearing, not generalizable |
| dstip | IP addresses - identity-bearing, not generalizable |
| sport | Port numbers - identity-bearing when used raw |
| dsport | Port numbers - identity-bearing when used raw |
| srcpt | Port numbers - identity-bearing when used raw |
| dstpt | Port numbers - identity-bearing when used raw |
| Ltime | Absolute timestamps - enable cross-flow tracking |
| Stime | Absolute timestamps - enable cross-flow tracking |
| attack_cat | Attack category - a label, not a feature |
| trans_depth | Application-layer parsing - requires payload access |
| response_body_len | Application-layer parsing - requires payload access |
| is_ftp_login | Application-layer parsing - requires payload access |
| ct_ftp_cmd | Application-layer parsing - requires payload access |
| ct_flw_http_mthd | Application-layer parsing - requires payload access |

## Policy Compliance Summary

- Features in FULL_METADATA: 37
- Features in RESTRICTED_METADATA: 17
- Rejected identity/label/timestamp fields: 9
- Rejected payload-derived fields: 5