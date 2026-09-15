import os, re, sys, psycopg2
from geopy.geocoders import Nominatim

body = os.environ.get('ISSUE_BODY', '')
def get_field(label):
    match = re.search(rf'### {label}\s*\n\s*(.*?)(?=\n###|\Z)', body, re.S)
    return match.group(1).strip() if match else ''

facility_type = get_field('Facility Type')
name = get_field('Facility Name')
street = get_field('Street Address')
city = get_field('City')
state = get_field('State Abbreviation').upper()
phone = get_field('Phone Number')
email = get_field('Email')

full_address = f'{street}, {city}, {state}'
geolocator = Nominatim(user_agent='animal_shelter_github_action')
loc = geolocator.geocode(full_address)
if not loc:
    print(f'Failed to geocode address: {full_address}')
    sys.exit(1)

table = 'shelters' if 'Shelter' in facility_type else 'clinics'
conn = psycopg2.connect(os.environ['DATABASE_URL'])
cur = conn.cursor()
query = f'''
    INSERT INTO {table} (name, address, email, state_code, phone, geom)
    VALUES (%s, %s, %s, %s, %s, ST_SetSRID(ST_MakePoint(%s, %s), 4326))
    ON CONFLICT (name, address) DO NOTHING;
'''
cur.execute(query, (name, full_address, email, state, phone, loc.longitude, loc.latitude))
conn.commit()
cur.close()
conn.close()
print(f'Successfully geocoded and inserted {name} into {table}.')