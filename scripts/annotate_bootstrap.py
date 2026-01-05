"""
Script to annotate bootstrap samples with labels.
"""
import json
import pandas as pd

# Load the bootstrap data
with open('data/bootstrap_export_20260105_010318.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Define label mappings based on discipline codes and keywords
PHASE_MAPPING = {
    'A': 'Administration',
    'B': 'Construction/Fabrication',
    'C': 'Commissioning',
    'D': 'Decommissioning', 
    'E': 'Engineering',
    'F': 'Feasibility',
    'G': 'General',
    'H': 'Hook-up',
    'I': 'Internal/Expenses',
    'J': 'Project Management',
    'K': 'FEED/Study',
    'L': 'Logistics',
    'M': 'Maintenance',
    'N': 'Installation',
    'O': 'Operations',
    'P': 'Procurement',
    'Q': 'Quality',
    'R': 'Risk',
    'S': 'Support',
    'T': 'Testing',
    'U': 'Utilities',
    'V': 'Verification',
    'W': 'Warranty',
    'X': 'Execution',
    'Y': 'Yard',
    'Z': 'Closeout'
}

DISCIPLINE_MAPPING = {
    'AA': 'Administration',
    'AB': 'Administration',
    'AC': 'Accounting',
    'AR': 'Architecture',
    'AU': 'Automation',
    'BB': 'Building',
    'BC': 'Building Construction',
    'BD': 'Building Design',
    'BE': 'Building Electrical',
    'BH': 'Building HVAC',
    'BM': 'Building Mechanical',
    'BP': 'Building Piping',
    'BS': 'Building Structural',
    'BZ': 'Multidiscipline',
    'CA': 'Cathodic Protection',
    'CC': 'Cost Control',
    'CE': 'Civil Engineering',
    'CI': 'Civil',
    'CM': 'Construction Management',
    'CO': 'Coating',
    'CP': 'Completion',
    'CS': 'Control Systems',
    'CV': 'Civil',
    'CW': 'Civil Works',
    'DC': 'Document Control',
    'DD': 'Detailed Design',
    'DM': 'Data Management',
    'EA': 'Electrical',
    'EC': 'Electrical Construction',
    'ED': 'Engineering Design',
    'EE': 'Electrical Engineering',
    'EI': 'Electrical/Instrumentation',
    'EL': 'Electrical',
    'EM': 'Environmental',
    'EN': 'Engineering',
    'EP': 'Electrical Power',
    'EQ': 'Equipment',
    'ES': 'Engineering Support',
    'ET': 'Electrical Telecom',
    'EX': 'Execution',
    'FA': 'Fabrication',
    'FD': 'FEED',
    'FE': 'Front End Engineering',
    'FI': 'Fire/Safety',
    'FM': 'Facilities Management',
    'FP': 'Fire Protection',
    'FR': 'Freight',
    'FS': 'Feasibility Study',
    'GA': 'General Administration',
    'GE': 'General Engineering',
    'GN': 'General',
    'GR': 'Grounding',
    'HC': 'Hook-up/Commissioning',
    'HE': 'Heat Exchanger',
    'HM': 'HAZMAT',
    'HS': 'Health & Safety',
    'HV': 'HVAC',
    'HW': 'Hardware',
    'HY': 'Hydraulics',
    'IA': 'Instrumentation/Automation',
    'IC': 'Instrument/Control',
    'ID': 'Instrument Design',
    'IE': 'Instrumentation Engineering',
    'IM': 'Information Management',
    'IN': 'Instrumentation',
    'IP': 'Instrument Piping',
    'IR': 'Insulation',
    'IS': 'Instrument Systems',
    'IT': 'IT Systems',
    'JC': 'Job Control',
    'KC': 'FEED Coordination',
    'KE': 'FEED Engineering',
    'KM': 'Knowledge Management',
    'KS': 'FEED Study',
    'LA': 'Layout',
    'LG': 'Logistics',
    'LI': 'Lifting',
    'LO': 'Logistics',
    'LP': 'Layout/Piping',
    'LS': 'Life Support',
    'MA': 'Marine',
    'MB': 'Mechanical Building',
    'MC': 'Mechanical Completion',
    'MD': 'Mechanical Design',
    'ME': 'Mechanical Engineering',
    'MH': 'Material Handling',
    'MI': 'Mining',
    'MK': 'Marketing',
    'ML': 'Materials',
    'MM': 'Mechanical Maintenance',
    'MO': 'Mobilization',
    'MP': 'Mechanical/Piping',
    'MR': 'Marine',
    'MS': 'Mechanical Systems',
    'MT': 'Material',
    'MW': 'Mechanical Works',
    'NA': 'Naval Architecture',
    'NC': 'Non-Conformance',
    'ND': 'NDT',
    'NM': 'Network Management',
    'OA': 'Office Administration',
    'OE': 'Operations Engineering',
    'OP': 'Operations',
    'OS': 'Offshore',
    'OW': 'Onshore Works',
    'PA': 'Process Automation',
    'PB': 'Prefabrication',
    'PC': 'Process Control',
    'PD': 'Process Design',
    'PE': 'Process Engineering',
    'PF': 'Prefab',
    'PG': 'Programming',
    'PI': 'Piping',
    'PJ': 'Project',
    'PL': 'Planning',
    'PM': 'Project Management',
    'PN': 'Piping Design',
    'PO': 'Procurement',
    'PP': 'Piping/Process',
    'PQ': 'Pre-Qualification',
    'PR': 'Procurement',
    'PS': 'Process Safety',
    'PT': 'Painting',
    'PU': 'Purchasing',
    'PV': 'Pressure Vessels',
    'PW': 'Power',
    'PX': 'Project Execution',
    'PY': 'Payload',
    'QA': 'Quality Assurance',
    'QC': 'Quality Control',
    'QM': 'Quality Management',
    'RA': 'Risk Assessment',
    'RC': 'Rotating Equipment',
    'RD': 'R&D',
    'RE': 'Rotating Equipment',
    'RF': 'Refurbishment',
    'RI': 'Risk',
    'RM': 'Risk Management',
    'RO': 'Rotating',
    'RP': 'Repair',
    'RS': 'Reservoir',
    'RT': 'Rotating',
    'RV': 'Review',
    'RW': 'Rework',
    'SA': 'Safety',
    'SB': 'Subsea',
    'SC': 'Scaffolding',
    'SD': 'Structural Design',
    'SE': 'Systems Engineering',
    'SF': 'Structural Fabrication',
    'SG': 'Staging',
    'SI': 'Site',
    'SK': 'Skid',
    'SL': 'Spool',
    'SM': 'Site Management',
    'SN': 'Structural Engineering',
    'SO': 'Site Operations',
    'SP': 'Support',
    'SQ': 'SQA',
    'SR': 'Structural',
    'SS': 'Subsea',
    'ST': 'Structural',
    'SU': 'Supervision',
    'SV': 'Survey',
    'SW': 'Software',
    'SY': 'Systems',
    'TA': 'Technical Assistance',
    'TB': 'Topsides',
    'TC': 'Telecom',
    'TD': 'Technical Documentation',
    'TE': 'Technical',
    'TF': 'Transformation',
    'TI': 'Tie-in',
    'TK': 'Tank',
    'TL': 'Tools',
    'TM': 'Transportation',
    'TN': 'Training',
    'TO': 'Topsides Operations',
    'TP': 'Topsides/Process',
    'TR': 'Transportation',
    'TS': 'Technical Support',
    'TT': 'Testing',
    'TU': 'Turnaround',
    'TW': 'Topside Works',
    'TX': 'Tax',
    'UA': 'Utilities Admin',
    'UC': 'Umbilical',
    'UD': 'Underground',
    'UE': 'Utilities Engineering',
    'UM': 'Utilities Mechanical',
    'UN': 'Utilities',
    'UP': 'Upgrade',
    'UT': 'Utilities',
    'VA': 'Valve',
    'VE': 'Verification',
    'VI': 'Vendor Inspection',
    'VM': 'Vendor Management',
    'VN': 'Vendor',
    'VP': 'Vessel/Pipe',
    'VR': 'Verification',
    'VS': 'Vessel',
    'VV': 'Vendor',
    'WA': 'Warranty',
    'WB': 'Wellbay',
    'WC': 'Welding Control',
    'WD': 'Well Design',
    'WE': 'Welding',
    'WH': 'Wellhead',
    'WI': 'Work Instructions',
    'WK': 'Workshop',
    'WL': 'Welding',
    'WM': 'Waste Management',
    'WO': 'Work Order',
    'WP': 'Work Package',
    'WR': 'Warehouse',
    'WS': 'Workshop',
    'WT': 'Weight',
    'WW': 'Welding Works',
    'XA': 'Execution Admin',
    'XE': 'Execution Engineering',
    'XM': 'Execution Management',
    'XP': 'Execution Planning',
    'XS': 'Execution Support',
    'YA': 'Yard Admin',
    'YD': 'Yard',
    'YM': 'Yard Management',
    'YO': 'Yard Operations',
    'YS': 'Yard Support',
    'ZA': 'Closeout Admin',
    'ZC': 'Closeout',
    'ZM': 'Closeout Management',
    'ZO': 'Closeout Operations',
    'I': 'Internal/Expenses',
    'B': 'Construction',
    'E': 'Engineering',
    'K': 'FEED/Study',
    'P': 'Procurement',
    'C': 'Commissioning'
}

# Work type keywords
WORK_TYPE_KEYWORDS = {
    'Design': ['design', 'drawing', 'model', 'layout', 'specification', 'spec', 'datasheet', 'diagram', 'schematic', 'p&id', 'pfd', 'isometric', '3d model', 'cad'],
    'Review': ['review', 'check', 'verify', 'validation', 'approval', 'idc', 'ifc', 'afr', 'afd', 'apd', 'comment', 'markup'],
    'Engineering': ['engineering', 'analysis', 'calculation', 'study', 'assessment', 'evaluation', 'optimization', 'sizing'],
    'Procurement': ['procurement', 'purchase', 'rfq', 'tender', 'bid', 'vendor', 'supplier', 'order', 'expediting', 'material'],
    'Fabrication': ['fabrication', 'manufacturing', 'prefab', 'assembly', 'welding', 'cutting', 'machining', 'shop'],
    'Construction': ['construction', 'installation', 'erection', 'mounting', 'laying', 'building', 'site work', 'field'],
    'Commissioning': ['commissioning', 'startup', 'start-up', 'testing', 'test', 'loop check', 'pre-comm', 'mc', 'rfo'],
    'Management': ['management', 'coordination', 'planning', 'scheduling', 'reporting', 'meeting', 'admin', 'supervision'],
    'Documentation': ['document', 'manual', 'procedure', 'report', 'dossier', 'handover', 'as-built'],
    'QA/QC': ['qa', 'qc', 'quality', 'inspection', 'ndt', 'audit', 'ncr'],
    'HSE': ['hse', 'safety', 'hazop', 'risk', 'permit', 'environmental'],
    'Support': ['support', 'assistance', 'service', 'maintenance', 'repair']
}

# Location keywords
LOCATION_KEYWORDS = {
    'Offshore': ['offshore', 'platform', 'fpso', 'jacket', 'topside', 'subsea', 'marine', 'vessel', 'rig', 'hook-up'],
    'Onshore': ['onshore', 'land', 'terminal', 'plant', 'refinery', 'facility', 'site', 'yard', 'shop', 'office', 'stavanger', 'bergen', 'oslo']
}

def get_phase_label(sample):
    prefix = sample.get('PHASE_PREFIX', '')
    desc = (sample.get('ACTIVITY_DESCRIPTION', '') or '').lower()
    sub_desc = (sample.get('SUB_PROJECT_DESCRIPTION', '') or '').lower()
    
    # Check description for phase keywords
    if any(kw in desc for kw in ['commissioning', 'startup', 'start-up', 'pre-comm', 'mc']):
        return 'Commissioning'
    if any(kw in desc for kw in ['engineering', 'design', 'study', 'feed', 'concept']):
        return 'Engineering'
    if any(kw in desc for kw in ['procurement', 'purchase', 'vendor', 'material']):
        return 'Procurement'
    if any(kw in desc for kw in ['construction', 'fabrication', 'installation', 'erection', 'prefab']):
        return 'Construction'
    if any(kw in desc for kw in ['hook-up', 'hookup', 'completion']):
        return 'Hook-up/Completion'
    
    # Fall back to prefix mapping
    return PHASE_MAPPING.get(prefix, 'General')

def get_discipline_label(sample):
    code = sample.get('DISCIPLINE_CODE', '')
    desc = (sample.get('ACTIVITY_DESCRIPTION', '') or '').lower()
    sub_desc = (sample.get('SUB_PROJECT_DESCRIPTION', '') or '').lower()
    
    # Check description for discipline keywords
    if any(kw in desc for kw in ['electrical', 'cable', 'power', 'lighting', 'switchgear']):
        return 'Electrical'
    if any(kw in desc for kw in ['instrument', 'control', 'automation', 'dcs', 'plc', 'scada']):
        return 'Instrumentation'
    if any(kw in desc for kw in ['piping', 'pipe', 'spool', 'valve', 'flange']):
        return 'Piping'
    if any(kw in desc for kw in ['structural', 'steel', 'structure', 'support']):
        return 'Structural'
    if any(kw in desc for kw in ['mechanical', 'equipment', 'rotating', 'pump', 'compressor', 'hvac']):
        return 'Mechanical'
    if any(kw in desc for kw in ['process', 'vessel', 'separator', 'heat exchanger']):
        return 'Process'
    if any(kw in desc for kw in ['civil', 'concrete', 'foundation', 'earthwork']):
        return 'Civil'
    if any(kw in desc for kw in ['telecom', 'communication', 'radio', 'paga']):
        return 'Telecom'
    if any(kw in desc for kw in ['subsea', 'umbilical', 'riser', 'flowline']):
        return 'Subsea'
    if any(kw in desc for kw in ['marine', 'naval', 'vessel', 'offshore']):
        return 'Marine'
    if any(kw in desc for kw in ['safety', 'fire', 'hazard', 'hse']):
        return 'Safety/HSE'
    if any(kw in desc for kw in ['coating', 'paint', 'insulation', 'fireproof']):
        return 'Coating/Insulation'
    if any(kw in desc for kw in ['document', 'admin', 'management', 'coordination']):
        return 'Project Management'
    
    # Fall back to code mapping
    return DISCIPLINE_MAPPING.get(code, DISCIPLINE_MAPPING.get(code[:1] if code else '', 'General'))

def get_work_type_label(sample):
    desc = (sample.get('ACTIVITY_DESCRIPTION', '') or '').lower()
    sub_desc = (sample.get('SUB_PROJECT_DESCRIPTION', '') or '').lower()
    combined = desc + ' ' + sub_desc
    
    for work_type, keywords in WORK_TYPE_KEYWORDS.items():
        if any(kw in combined for kw in keywords):
            return work_type
    
    return 'General'

def get_location_label(sample):
    desc = (sample.get('ACTIVITY_DESCRIPTION', '') or '').lower()
    project_desc = (sample.get('PROJECT_DESCRIPTION', '') or '').lower()
    project_name = (sample.get('PROJECT_NAME', '') or '').lower()
    combined = desc + ' ' + project_desc + ' ' + project_name
    
    for location, keywords in LOCATION_KEYWORDS.items():
        if any(kw in combined for kw in keywords):
            return location
    
    # Default based on common patterns
    return 'Not Specified'

# Annotate all samples
for sample in data['samples']:
    sample['PHASE_LABEL'] = get_phase_label(sample)
    sample['DISCIPLINE_LABEL'] = get_discipline_label(sample)
    sample['WORK_TYPE_LABEL'] = get_work_type_label(sample)
    sample['LOCATION_LABEL'] = get_location_label(sample)
    sample['NOTES'] = ''

# Save annotated data
output_path = 'data/bootstrap_annotated.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

# Also save as CSV for training
df = pd.DataFrame(data['samples'])
df.to_csv('data/bootstrap_annotated.csv', index=False, encoding='utf-8-sig')

print(f"Annotated {len(data['samples'])} samples")
print(f"Saved to: {output_path}")
print(f"Saved CSV to: data/bootstrap_annotated.csv")
print()
print("Label Distribution:")
print("=" * 60)
print()

print("PHASE_LABEL:")
phase_counts = {}
for s in data['samples']:
    label = s['PHASE_LABEL']
    phase_counts[label] = phase_counts.get(label, 0) + 1
for label, count in sorted(phase_counts.items(), key=lambda x: -x[1]):
    print(f"  {label}: {count}")

print()
print("DISCIPLINE_LABEL:")
disc_counts = {}
for s in data['samples']:
    label = s['DISCIPLINE_LABEL']
    disc_counts[label] = disc_counts.get(label, 0) + 1
for label, count in sorted(disc_counts.items(), key=lambda x: -x[1]):
    print(f"  {label}: {count}")

print()
print("WORK_TYPE_LABEL:")
work_counts = {}
for s in data['samples']:
    label = s['WORK_TYPE_LABEL']
    work_counts[label] = work_counts.get(label, 0) + 1
for label, count in sorted(work_counts.items(), key=lambda x: -x[1]):
    print(f"  {label}: {count}")

print()
print("LOCATION_LABEL:")
loc_counts = {}
for s in data['samples']:
    label = s['LOCATION_LABEL']
    loc_counts[label] = loc_counts.get(label, 0) + 1
for label, count in sorted(loc_counts.items(), key=lambda x: -x[1]):
    print(f"  {label}: {count}")

print()
print("=" * 60)
print("Sample annotations:")
print("=" * 60)
for i, s in enumerate(data['samples'][:10], 1):
    print(f"\n{i}. {s['ACTIVITY_DESCRIPTION'][:50]}...")
    print(f"   Phase: {s['PHASE_LABEL']}, Discipline: {s['DISCIPLINE_LABEL']}")
    print(f"   Work Type: {s['WORK_TYPE_LABEL']}, Location: {s['LOCATION_LABEL']}")
