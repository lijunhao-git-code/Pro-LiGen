"""Original Chinese feedback formatter; no API calls."""
import json
def optimization_feedback(records, entry):
    proposals = [dict(position=r['candidate_position'], canonical=r.get('canonical'),
                      valid=r['valid'], eligible=r['eligible'], reason=r['reason'],
                      RC=r.get('RC'), QED=r.get('QED'), SA_raw=r.get('SA_raw')) for r in records]
    return ('\n【程序实际计算的上一轮结果】\n'+json.dumps(proposals, ensure_ascii=False)+
            '\n以上由 RDKit 和标准 SA scorer 实际计算。未计算对接。请保持环数恰好为 '+
            str(entry['target_rc'])+'；所有有效且满足条件的上一轮候选会保留到最终程序排序。')
