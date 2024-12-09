use csv::ReaderBuilder;
use ndarray::{Array2, Array1, s};
use std::error::Error;
use std::fmt;


// Defines ColumnVal for holding country names and scores
#[derive(Debug, Clone)]
pub enum ColumnVal {
    Country(String),
    Score(f64),
}

// Implementing Display for ColumnVal
impl fmt::Display for ColumnVal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ColumnVal::Country(val) => write!(f, "{}", val),  // Displays country name
            ColumnVal::Score(val) => write!(f, "{:.2}", val), // Displays score with 2 decimals
        }
    }
}

// Define DataFrame structure for holding labels, types, and rows
#[derive(Debug)]
pub struct DataFrame {
    pub labels: Vec<String>,
    pub types: Vec<u32>,
    pub rows: Vec<Vec<ColumnVal>>,
}

impl DataFrame {
    // Initialized the DataFrame
    pub fn new() -> Self {
        DataFrame {
            labels: Vec::new(),
            types: Vec::new(),
            rows: Vec::new(),
        }
    }
    // Reads the CSV file and turns it into a vector of values
    pub fn read_csv(&mut self, path: &str, types: &Vec<u32>) -> Result<(), Box<dyn Error>> {
        let mut rdr = ReaderBuilder::new()
            .delimiter(b',')
            .has_headers(true) // Assumes the first row is headers
            .flexible(true)
            .from_path(path)?;

        let mut first_row = true;
        for result in rdr.records() {
            let r = result?;
            let mut row = Vec::new();

            if first_row {
                self.labels = r.iter().map(|s| s.to_string()).collect();
                first_row = false;
                continue;
            }

            if r.len() != types.len() {
                return Err(format!(
                    "Row has {} columns, expected {}",
                    r.len(),
                    types.len()
                )
                .into());
            }

            for (i, elem) in r.iter().enumerate() {
                match types[i] {
                    1 => row.push(ColumnVal::Country(elem.to_string())),
                    2 => {
                        let value = elem.parse::<f64>().unwrap_or(0.0);
                        row.push(ColumnVal::Score(value));
                    }
                    _ => return Err(format!("Unknown type: {}", types[i]).into()),
                }
            }
            self.rows.push(row);
        }

        self.types = types.clone();
        Ok(())
    }

    // Gets data from the other columns and puts it in vectors as an f64 value
    pub fn get_scores(&self) -> Vec<Array1<f64>> {
        self.rows
            .iter()
            .map(|row| {
                row.iter()
                    .filter_map(|column_value| match column_value {
                        ColumnVal::Score(val) => Some(*val),
                        _ => None,
                    })
                    .collect::<Vec<f64>>()
            })
            .map(|v| Array1::from(v))
            .collect()
    }
}