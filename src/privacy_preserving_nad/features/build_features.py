"""Feature engineering and feature policy documentation for UNSW-NB15 dataset.

This module defines the metadata-only feature policy and documents why each
selected feature qualifies as traffic metadata (observable without packet
payload inspection).
"""

from typing import Dict, List, Set
import yaml
from pathlib import Path


# =============================================================================
# FEATURE POLICY DOCUMENTATION
# =============================================================================

FEATURE_POLICY = """
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
"""

# Feature set definitions (mirror config.yaml)
FULL_METADATA_FEATURES = [
    "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate",
    "sttl", "dttl", "sload", "dload", "sloss", "dloss",
    "sinpkt", "dinpkt", "sjit", "djit", "swin", "dwin",
    "stcpb", "dtcpb", "tcprtt", "synack", "ackdat",
    "smean", "dmean",
    "ct_srv_src", "ct_state_ttl", "ct_dst_ltm", "ct_src_dport_ltm",
    "ct_dst_sport_ltm", "ct_dst_src_ltm", "ct_src_ltm", "ct_srv_dst",
    "is_sm_ips_ports", "proto", "service", "state"
]

RESTRICTED_METADATA_FEATURES = [
    "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate",
    "sttl", "dttl", "sload", "dload", "sinpkt", "dinpkt",
    "sjit", "djit", "tcprtt", "synack", "ackdat"
]

CATEGORICAL_FEATURES = ["proto", "service", "state"]

EXCLUDED_FEATURES = [
    "srcip", "dstip", "sport", "dsport", "srcpt", "dstpt",
    "Ltime", "Stime", "attack_cat"
]

# Rejected because they can only be produced by parsing application content.
PAYLOAD_DERIVED_FEATURES = [
    "trans_depth", "response_body_len", "is_ftp_login",
    "ct_ftp_cmd", "ct_flw_http_mthd"
]

# Documentation for each feature
FEATURE_DOCUMENTATION = {
    "dur": "Flow duration in seconds. Observable from flow start/end timestamps in packet headers.",
    "spkts": "Source-to-destination packet count. Countable from packet headers.",
    "dpkts": "Destination-to-source packet count. Countable from packet headers.",
    "sbytes": "Source-to-destination byte count. Sum of IP packet lengths from headers.",
    "dbytes": "Destination-to-source byte count. Sum of IP packet lengths from headers.",
    "rate": "Packets per second (spkts+dpkts)/dur. Derived from counts and duration.",
    "sttl": "Source TTL value from IP header. Directly observable.",
    "dttl": "Destination TTL value from IP header. Directly observable.",
    "sload": "Source-to-destination bit rate. Derived from sbytes and dur.",
    "dload": "Destination-to-source bit rate. Derived from dbytes and dur.",
    "sloss": "Source packet loss/retransmissions. Inferred from TCP sequence numbers.",
    "dloss": "Destination packet loss/retransmissions. Inferred from TCP sequence numbers.",
    "sinpkt": "Source inter-packet arrival time (mean). From packet timestamps.",
    "dinpkt": "Destination inter-packet arrival time (mean). From packet timestamps.",
    "sjit": "Source jitter (variance in inter-arrival). From packet timestamps.",
    "djit": "Destination jitter. From packet timestamps.",
    "swin": "Source TCP window size. From TCP header.",
    "dwin": "Destination TCP window size. From TCP header.",
    "stcpb": "Source TCP sequence number base. From TCP header.",
    "dtcpb": "Destination TCP sequence number base. From TCP header.",
    "tcprtt": "TCP round-trip time from handshake. From SYN/SYN-ACK timestamps.",
    "synack": "Time between SYN and SYN-ACK. From packet timestamps.",
    "ackdat": "Time between SYN-ACK and ACK. From packet timestamps.",
    "smean": "Mean packet size (source->dest). sbytes/spkts.",
    "dmean": "Mean packet size (dest->source). dbytes/dpkts.",
    "trans_depth": "Application-layer transaction depth. Requires HTTP parsing - excluded as payload-derived.",
    "response_body_len": "HTTP response body length. Application header inside TLS - excluded as payload-derived.",
    "ct_srv_src": "Connection count: same service, same source (time window). Behavioral aggregate.",
    "ct_state_ttl": "Connection count: same state, same TTL. Behavioral aggregate.",
    "ct_dst_ltm": "Connection count: same destination (last time window). Behavioral aggregate.",
    "ct_src_dport_ltm": "Connection count: same source, same dest port (last time window). Aggregate count only; ports never enter the model.",
    "ct_dst_sport_ltm": "Connection count: same dest, same src port (last time window). Aggregate count only; ports never enter the model.",
    "ct_dst_src_ltm": "Connection count: same dest and src pair (last time window). Keyed on header addresses; only the count is used.",
    "is_ftp_login": "FTP login flag. Requires FTP command parsing - excluded as payload-derived.",
    "ct_ftp_cmd": "FTP command counter. Requires FTP command parsing - excluded as payload-derived.",
    "ct_flw_http_mthd": "HTTP method counter. Requires HTTP request parsing - excluded as payload-derived.",
    "ct_src_ltm": "Connection count: same source (last time window). Behavioral aggregate.",
    "ct_srv_dst": "Connection count: same service, same destination. Behavioral aggregate.",
    "is_sm_ips_ports": "Same source/dest IP and port indicator. Aggregate derived from headers; raw addresses never enter the model.",
    "proto": "Transport protocol (tcp/udp/icmp). From IP header protocol field.",
    "service": "Application service label. Assigned by capture tooling from protocol/port mapping, obtainable from headers without payload.",
    "state": "Connection state (FIN/CON/REQ/RST/etc.). From TCP flags in headers.",
}


def get_feature_set(feature_set_name: str) -> List[str]:
    """Get feature list for a named feature set."""
    if feature_set_name == "FULL_METADATA":
        return FULL_METADATA_FEATURES.copy()
    elif feature_set_name == "RESTRICTED_METADATA":
        return RESTRICTED_METADATA_FEATURES.copy()
    else:
        raise ValueError(f"Unknown feature set: {feature_set_name}")


def get_feature_documentation(feature: str) -> str:
    """Get documentation for a specific feature."""
    return FEATURE_DOCUMENTATION.get(feature, "No documentation available.")


def validate_feature_policy(df_columns: List[str]) -> Dict:
    """Validate that dataset columns comply with feature policy."""
    df_columns_set = set(df_columns)

    # Check for excluded features that are present
    excluded_present = [f for f in EXCLUDED_FEATURES if f in df_columns_set]

    # Check which allowed features are available
    full_available = [f for f in FULL_METADATA_FEATURES if f in df_columns_set]
    full_missing = [f for f in FULL_METADATA_FEATURES if f not in df_columns_set]

    restricted_available = [f for f in RESTRICTED_METADATA_FEATURES if f in df_columns_set]
    restricted_missing = [f for f in RESTRICTED_METADATA_FEATURES if f not in df_columns_set]

    return {
        "excluded_features_present": excluded_present,
        "full_metadata_available": full_available,
        "full_metadata_missing": full_missing,
        "restricted_metadata_available": restricted_available,
        "restricted_metadata_missing": restricted_missing,
        "policy_compliant": len(excluded_present) == 0
    }


def generate_feature_report(config_path: str = "configs/config.yaml") -> str:
    """Generate a markdown report documenting the feature policy."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    report = []
    report.append("# Feature Policy Documentation\n")
    report.append(FEATURE_POLICY)
    report.append("\n## Feature Set Definitions\n")

    for fs_name in ["FULL_METADATA", "RESTRICTED_METADATA"]:
        features = config["feature_sets"][fs_name]
        report.append(f"\n### {fs_name} ({len(features)} features)\n")
        report.append("| Feature | Description | Category |")
        report.append("|---------|-------------|----------|")

        for f in features:
            doc = FEATURE_DOCUMENTATION.get(f, "No documentation")
            # Determine category
            if f in CATEGORICAL_FEATURES:
                cat = "Categorical"
            elif f in ["dur", "sinpkt", "dinpkt", "sjit", "djit", "tcprtt", "synack", "ackdat"]:
                cat = "Timing"
            elif f in ["spkts", "dpkts", "sbytes", "dbytes", "rate", "sload", "dload"]:
                cat = "Volume/Rate"
            elif f in ["sttl", "dttl", "swin", "dwin", "stcpb", "dtcpb", "sloss", "dloss"]:
                cat = "TCP State"
            elif f in ["smean", "dmean", "trans_depth", "response_body_len"]:
                cat = "Packet Size"
            elif f.startswith("ct_") or f.startswith("is_"):
                cat = "Behavioral Aggregate"
            else:
                cat = "Other"
            report.append(f"| {f} | {doc} | {cat} |")

    report.append("\n## Excluded Features\n")
    report.append("| Feature | Reason for Exclusion |")
    report.append("|---------|---------------------|")
    for f in EXCLUDED_FEATURES:
        if f in ["srcip", "dstip"]:
            reason = "IP addresses - identity-bearing, not generalizable"
        elif f in ["sport", "dsport", "srcpt", "dstpt"]:
            reason = "Port numbers - identity-bearing when used raw"
        elif f in ["Ltime", "Stime"]:
            reason = "Absolute timestamps - enable cross-flow tracking"
        elif f == "attack_cat":
            reason = "Attack category - a label, not a feature"
        else:
            reason = "Not traffic metadata"
        report.append(f"| {f} | {reason} |")
    for f in PAYLOAD_DERIVED_FEATURES:
        report.append(f"| {f} | Application-layer parsing - requires payload access |")

    report.append("\n## Policy Compliance Summary\n")
    report.append(f"- Features in FULL_METADATA: {len(FULL_METADATA_FEATURES)}")
    report.append(f"- Features in RESTRICTED_METADATA: {len(RESTRICTED_METADATA_FEATURES)}")
    report.append(f"- Rejected identity/label/timestamp fields: {len(EXCLUDED_FEATURES)}")
    report.append(f"- Rejected payload-derived fields: {len(PAYLOAD_DERIVED_FEATURES)}")

    return "\n".join(report)


def main():
    """Generate feature policy report."""
    report = generate_feature_report()
    output_path = Path("docs/feature_policy.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)
    print(f"Feature policy report saved to: {output_path}")


if __name__ == "__main__":
    main()