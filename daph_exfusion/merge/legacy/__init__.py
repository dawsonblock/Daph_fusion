"""Legacy sparse merge operators (v2.5 archive).

These operators (DARE, TIES, DARE-TIES, TIES-Fisher, DARE-TIES-Fisher)
are retained as controlled baselines for comparison against the v3 dense
merge trunk. They are NOT part of the mainline.

Import from here for benchmark purposes only:
    from daph_exfusion.merge.legacy import op_dare, op_ties, op_dare_ties
"""
from daph_exfusion.merge.legacy.dare import op_dare
from daph_exfusion.merge.legacy.ties import op_ties, _ties_trim
from daph_exfusion.merge.legacy.dare_ties import op_dare_ties

__all__ = ["op_dare", "op_ties", "op_dare_ties", "_ties_trim"]
