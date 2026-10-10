"""PRISM parameter counts printed in Section 3.2 and Appendix A."""
from emc import models
from emc.models.prism import count_parameters, dormant_parameters


def test_prism_parameter_counts():
    m = models.build("prism", "prism_dn")
    assert count_parameters(m, active=True) == 40401
    assert count_parameters(m) == 44465
    dn = sum(p.numel() for n, p in m.named_parameters()
             if n.startswith("encoder.divnorm.") and n not in dormant_parameters(m))
    assert dn == 99


def test_subtractive_control_matches_active_parameters():
    """A2 (ng_sub) is matched to the complete model in active parameter count."""
    assert count_parameters(models.build("prism", "ng_sub"), active=True) == \
        count_parameters(models.build("prism", "prism_dn"), active=True)
