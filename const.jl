# TOKENS
PAD_TOKEN = "<PAD>"
PAD_ID = 0
UNK_TOKEN = "<UNK>"
UNK_ID = 1

# hardcoded token embeddings
PAD_EMB = rand(Float32, 300)
UNK_EMB = rand(Float32, 300)
SUBJ_EMB = rand(Float32, 300)
OBJ_EMB = rand(Float32, 300)

# SUBJ & OBJ NER
SUBJ_NER_TO_ID = Dict(PAD_TOKEN => 0, UNK_TOKEN => 1, "ORGANIZATION" => 2, "PERSON" => 3)

OBJ_NER_TO_ID = Dict(PAD_TOKEN => 0, UNK_TOKEN => 1, "PERSON" => 2, "ORGANIZATION" => 3, "DATE" => 4, "NUMBER" => 5, "TITLE" => 6,
    "COUNTRY" => 7, "LOCATION" => 8, "CITY" => 9, "MISC" => 10, "STATE_OR_PROVINCE" => 11, "DURATION" => 12, "NATIONALITY" => 13,
    "CAUSE_OF_DEATH" => 14, "CRIMINAL_CHARGE" => 15, "RELIGION" => 16, "URL" => 17, "IDEOLOGY" => 18)

LABEL_TO_ID = Dict("no_relation" => 0, "per:title" => 1, "org:top_members/employees" => 2, "per:employee_of" => 3, "org:alternate_names" => 4,
    "org:country_of_headquarters" => 5, "per:countries_of_residence" => 6, "org:city_of_headquarters" => 7, "per:cities_of_residence" => 8,
    "per:age" => 9, "per:stateorprovinces_of_residence" => 10, "per:origin" => 11, "org:subsidiaries" => 12, "org:parents" => 13,
    "per:spouse" => 14, "org:stateorprovince_of_headquarters" => 15, "per:children" => 16, "per:other_family" => 17, "per:alternate_names" => 18,
    "org:members" => 19, "per:siblings" => 20, "per:schools_attended" => 21, "per:parents" => 22, "per:date_of_death" => 23, "org:member_of" => 24,
    "org:founded_by" => 25, "org:website" => 26, "per:cause_of_death" => 27, "org:political/religious_affiliation" => 28, "org:founded" => 29,
    "per:city_of_death" => 30, "org:shareholders" => 31, "org:number_of_employees/members" => 32, "per:date_of_birth" => 33, "per:city_of_birth" => 34,
    "per:charges" => 35, "per:stateorprovince_of_death" => 36, "per:religion" => 37, "per:stateorprovince_of_birth" => 38, "per:country_of_birth" => 39,
    "org:dissolved" => 40, "per:country_of_death" => 41)

ID_TO_LABEL = Dict(v => k for (k, v) in LABEL_TO_ID)
