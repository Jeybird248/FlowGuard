"""FlowGuard: cross-modal information-flow anomaly detection for MLLMs."""
from flowguard.flowvectors import FlowVector, compute_flowvector
from flowguard.detector import FlowGuardDetector

__all__ = ["FlowVector", "compute_flowvector", "FlowGuardDetector"]
__version__ = "0.1.0"
