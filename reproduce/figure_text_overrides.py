"""Current editorial annotations, without editing archived plotting code or data."""
import hashlib
import json

def coordinate_snapshot(fig):
    result=[]
    for ax in fig.axes:
        row=dict(xlim=list(ax.get_xlim()),ylim=list(ax.get_ylim()),collections=[],lines=[])
        for item in ax.collections:
            if hasattr(item,'get_offsets'):row['collections'].append(item.get_offsets().tolist())
            if hasattr(item,'get_segments'):row['collections'].append([x.tolist() for x in item.get_segments()])
        for item in ax.lines:
            row['lines'].append([list(item.get_xdata()),list(item.get_ydata())])
        result.append(row)
    return json.dumps(result,sort_keys=True,default=float)

def apply_editorial_labels(renderer):
    original_save=renderer.save
    receipt=dict(changed_text_count=0,numeric_coordinates_unchanged=False,figures={})
    # Original renderer and scientific inputs remain frozen. Only text artists change.
    replacements={
        'fig1_current_pipeline':[
            ('Shared five-turn dialogue','Two-round generation'),
            ('Turns 1–3: contextual reasoning\nTurn 4: 3 proposal positions\nTurn 5: 3 proposal positions',
             'Context-guided proposals\nRound 1: up to 3 candidates\nRound 2: up to 3 candidates'),
            ('Compute after turn 4','Computed feedback'),
            ('RDKit: parse, fragments, RC\nQED, SA and exact-copy flags\nSend computed turn-5 feedback',
             'Check ring count and eligibility\nCompute QED and SA\nGuide the second round'),
            ('turn 4','round 1'),
            ('Retain qualified union','Retain qualified pool'),
            ('Turns 4 + 5: up to 6 positions\nSingle fragment; exact target RC\nNo exact seed/reference copy',
             'Combine both proposal rounds\nRequired RC; single fragment\nExclude seed and example copies'),
            ('Deterministic delivery','Select final molecule'),
            ('Max QED in qualified union\nSA breaks equal-QED ties\nEmpty union → no delivery',
             'Choose highest QED\nReturn one molecule per run\nNo qualified candidate: no output'),
            ('Delivery S and QED yield U\nTarget-paired comparisons\nCalibrated docking afterward',
             'Delivery rate and QED yield\nTarget-level comparisons\nSeparate docking evaluation'),
            ('turns 4 + 5 candidates','candidates from both rounds'),
            ('Invalid / missing tags consume positions. Bounded technical retries; no quality-driven resampling.\n'
             'Vina is separate from generation/selection. Current Study B is distinct from historical Study A.',None),
        ],
        'fig2_historical_tradeoffs':[
            ('(a) Historical strategy response','(a) Guidance-strategy profiles'),
            ('Study A: 25 target means / group; 500 retained historical records','Study A: 25 targets; 500 outputs'),
        ],
        'fig3_mechanism':[
            ('(a) New paired fifth-turn branches','(a) Paired additional-feedback comparison'),
            ('Turn 4\nQED','Round 1\nQED'),
            ('Turn 5\nQED','Round 2\nQED'),
            ('Budget: 3 / 3 / 6 / 6 positions\nSaved attempts; not new ablation',
             'Candidates per policy: 3 / 3 / 6 / 6\nReplay of saved generation outputs'),
        ],
        'fig4_downstream':[
            ('pH: original family-specific Holm p\nNonsignificant ≠ equivalent',
             'pH: family-specific Holm-adjusted p'),
        ],
    }
    def save(fig,name,output):
        if name in replacements:
            from matplotlib.text import Text
            fig.canvas.draw()
            before=coordinate_snapshot(fig)
            labels=fig.findobj(match=Text)
            edits=[]
            normalize=lambda text:' '.join(text.split())
            for old,new in replacements[name]:
                matches=[label for label in labels if label.get_visible() and normalize(label.get_text())==normalize(old)]
                assert len(matches)==1,(name,old,'Expected exactly one original text artist')
                target=matches[0]
                target.set_text(new or '')
                # Tick text is recreated from its formatter at save time. Update that
                # formatter too, without changing tick locations or any data artist.
                for ax in fig.axes:
                    ticks=ax.get_xticklabels()
                    if any(label is target for label in ticks):
                        ax.set_xticklabels([new if label is target else label.get_text() for label in ticks])
                if new is None:target.set_visible(False)
                edits.append(dict(old=old,new=new))
            after=coordinate_snapshot(fig)
            assert before==after,'An editorial change must not change any numeric coordinate'
            receipt['figures'][name]=dict(changed_text_count=len(edits),edits=edits,
                numeric_coordinates_unchanged=True,
                numerical_coordinate_sha256=hashlib.sha256(before.encode()).hexdigest())
            receipt['changed_text_count']+=len(edits)
            receipt['numeric_coordinates_unchanged']=True
        original_save(fig,name,output)
    renderer.save=save
    return receipt
