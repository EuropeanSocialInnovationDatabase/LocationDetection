CREATE TABLE IF NOT EXISTS "location_info" (
	"loc_id"	INTEGER NOT NULL,
	"loc_name"	TEXT NOT NULL UNIQUE,
	"loc_info"	TEXT,
	PRIMARY KEY("loc_id" AUTOINCREMENT)
)