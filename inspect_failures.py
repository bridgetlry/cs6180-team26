from aci_data_loader import load_train

train = load_train()

failing_ids = {"23-aci", "27-aci", "30-aci", "31-aci", "34-aci"}

for enc in train:
    if enc.encounter_id in failing_ids:
        print(f"\n{'='*60}")
        print(f"ID: {enc.encounter_id}")
        print(f"CC (metadata): {enc.chief_complaint_gt}")
        print(f"\nTRANSCRIPT (first 500 chars):")
        print(enc.transcript[:500])
        print(f"\nREFERENCE NOTE (first 300 chars):")
        print(enc.reference_note[:300])