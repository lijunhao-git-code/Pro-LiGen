# API reproducibility supplement

The CSV files contain only whitelisted request metadata, usage, cost estimates and timings. Full responses, credentials and literature text are excluded. The original scientific observations and selections are unchanged.

`model_settings.json` records caller identifiers, submitted settings, dates and provider-returned metadata; version labels follow the archived provider documentation. A compatibility alias is not a pinned checkpoint. `resource_definitions.json` defines time and cost accounting.

`ring_prompt_templates_worked.json` preserves the sent Chinese instructions, with literature/binding-record values marked as input placeholders. Original seed/reference structures are retained for a fixed worked task. For R, substitute its recorded references into the same M template.

`continuous_prompt_builders.py` preserves the Chinese design, refinement and feedback functions. Entries supply property (logp/tpsa), uniprotid, chemblid, seed_smiles, seed_property, target_lower, target_upper and exemplars. RDKit is required only to format reference descriptors. In the continuous-property experiments, the first context message replaces the ring-count definition with the external RDKit property definition; turns 2/3 use the same supplied literature/target context. `prompt_template_sources.json` preserves the exact source-function text and hashes. `ring_feedback_builder.py` formats the recorded ring feedback; pass target_rc and proposal records.

`feedback_C_messages75.json` provides every archived base fifth-turn prompt and computation suffix, without the literature-containing conversation prefix. Study C sends either the base prompt alone or base plus its suffix. The same earlier dialogue is used in each pair. The original Study-A definitions remain in data/SI_records/A_stored_prompt_definitions_*.json, linked by A_protocol_provenance.json.

To use the builders: import the module, populate entry from the released task/reference records, build the turn-4 and turn-5 messages, and append feedback computed from actual candidates. No model call is needed to inspect or verify the templates. The English SI text is a reading guide; Chinese source strings define the sent instructions.
