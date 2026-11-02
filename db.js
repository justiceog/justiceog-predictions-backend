const Database = require('better-sqlite3');
const db = new Database('predictions.db');

// Initialize database tables
db.exec(`
  CREATE TABLE IF NOT EXISTS matches (
    id TEXT PRIMARY KEY,
    league TEXT,
    home_team TEXT,
    away_team TEXT,
    hx REAL,
    ax REAL,
    quality REAL,
    odds_json TEXT,
    sub_text TEXT,
    kickoff TEXT,
    venue TEXT,
    referee TEXT,
    updated_at INTEGER
  );

  CREATE TABLE IF NOT EXISTS settled_bets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id TEXT,
    match_name TEXT,
    league TEXT,
    market TEXT,
    odds REAL,
    model_prob REAL,
    ev REAL,
    score TEXT,
    outcome INTEGER,
    profit REAL,
    settled_at TEXT
  );
`);

module.exports = db;
