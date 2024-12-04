use csv::ReaderBuilder;
use ndarray::{Array2, Array1, s};
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};
use ndarray_rand::RandomExt;
use std::error::Error;
use std::fmt;
use std::path::Path;

// Define ColumnVal for holding different types of data (e.g., country names, scores)
#[derive(Debug, Clone)]
enum ColumnVal {
    Country(String),
    Score(f64),
}

// Implementing Display for ColumnVal
impl fmt::Display for ColumnVal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ColumnVal::Country(val) => write!(f, "{}", val),  // Display country name
            ColumnVal::Score(val) => write!(f, "{:.2}", val), // Display score with 2 decimals
        }
    }
}

// Define DataFrame structure for holding labels, types, and rows
#[derive(Debug)]
pub struct DataFrame {
    labels: Vec<String>,
    types: Vec<u32>,
    rows: Vec<Vec<ColumnVal>>,
}

impl DataFrame {
    pub fn new() -> Self {
        DataFrame {
            labels: Vec::new(),
            types: Vec::new(),
            rows: Vec::new(),
        }
    }

    pub fn read_csv(&mut self, path: &str, types: &Vec<u32>) -> Result<(), Box<dyn Error>> {
        let mut rdr = ReaderBuilder::new()
            .delimiter(b',')
            .has_headers(true) // Assuming the first row is headers
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