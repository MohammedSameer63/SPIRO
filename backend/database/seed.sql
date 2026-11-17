INSERT INTO waste_categories (name, description)
VALUES
    ('Recoverable', 'Clean recyclable materials — plastic, glass, metal, paper, cardboard'),
    ('Organic', 'Biodegradable food, garden, and natural waste — compost or green bin'),
    ('Non-Recoverable', 'Contaminated, composite, or residual waste — general waste bin'),
    ('Hazardous', 'Dangerous chemicals, flammables, and sharp metal — hazardous collection point'),
    ('Sanitary', 'Medical, hygiene, and bodily waste — sealed sanitary bin'),
    ('E-Waste', 'Electronics and batteries — e-waste collection point'),
    ('Reject', 'Unclassified or heavily mixed waste requiring manual inspection')
ON CONFLICT (name) DO NOTHING;

INSERT INTO wards (
    name,
    zone,
    description
)
VALUES
(
    'Ward 1',
    'North Zone',
    'Development Ward'
),
(
    'Ward 2',
    'South Zone',
    'Testing Ward'
),
(
    'Ward 3',
    'East Zone',
    'Production Demo'
);
