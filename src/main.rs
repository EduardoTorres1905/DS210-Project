use std::cmp::Ordering;
use std::fs::File;
use std::io::{self, BufRead};
use std::error::Error;
use ndarray::{Array1, Array2};
use std::collections::HashMap;

mod DataFrame;
use DataFrame::DataFrame as OtherDataFrame;

use crate::DataFrame::ColumnVal;
use ndarray::array;

mod Country_Score_Creator;
use Country_Score_Creator::{euclidean_distance, create_similarity_matrix, simple_clustering};
// Main function: loads data, compute similarity, and apply clustering

fn main() -> Result<(), Box<dyn Error>> {
    // Create a new DataFrame
    let mut df = OtherDataFrame::new();

    // Define column types (1 for country name, 2 for numeric score)
    let types = vec![1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2];

    // Read the CSV file
    df.read_csv("most-dangerous-countries-for-women-2024.csv", &types)?;

    // Extract scores (using the Women Peace and Security Index Score as the primary metric)
    let scores: Vec<f64> = df.rows.iter()
        .map(|row| {
            match &row[1] {
                ColumnVal::Score(val) => *val,
                _ => 0.0  // Default to 0 if no score
            }
        })
        .collect();

    // Get scores for similarity matrix computation
    let score_arrays = df.get_scores();

    // Create similarity matrix
    let similarity_matrix = create_similarity_matrix(&score_arrays);

    // Perform clustering
    let clusters = simple_clustering(&similarity_matrix, scores.clone());

    // Prepare results based on score ranges
    let mut clustered_countries: HashMap<String, Vec<(String, f64)>> = HashMap::new();

    // Classify countries based on their score into one of three categories
    for (i, score) in scores.iter().enumerate() {
        let country = match &df.rows[i][0] {
            ColumnVal::Country(name) => name.clone(),
            _ => continue,
        };
        
        // Classify based on the score ranges
        let category = if *score <= 0.60 {
            "Most Dangerous Countries"
        } else if *score <= 0.80 {
            "Moderately Safe Countries"
        } else {
            "Safest Countries"
        };

        clustered_countries.entry(category.to_string())
            .or_insert_with(Vec::new)
            .push((country, *score));
    }

    println!("Clustering of Countries by Danger to Women:");

    // Mapping cluster categories to their names
    let cluster_names = ["Most Dangerous Countries", "Moderately Safe Countries", "Safest Countries"];
    
    for name in cluster_names.iter() {
        println!("\n{}:", name);
        if let Some(countries) = clustered_countries.get(*name) {
            // Sort the countries differently depending on the category
            let mut sorted_countries = countries.clone();
            
            if *name == "Most Dangerous Countries" {
                sorted_countries.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            } else {
                sorted_countries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            }

            // Take the top 5 countries
            let top_5 = sorted_countries.into_iter().take(5);
            for (country, score) in top_5 {
                println!("- {} (Score: {:.3})", country, score);
            }
        }
    }

    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euclidean_distance() {
        // Example inputs
        let a = ndarray::array![4.0, 3.0, 2.0, 1.0];
        let b = ndarray::array![1.0, 2.0, 3.0, 4.0];

        // Compute the distance
        let distance = euclidean_distance(&a, &b);
        let rounded = (distance * 100.0).round() / 100.0;

        // Compare the distance with the expected value
        assert_eq!(rounded, 4.47);
    }

    #[test]
    fn test_min_max() -> io::Result<()> {
        // Define the file path
        let file_path = "most-dangerous-countries-for-women-2024.csv";

        // Open the file
        let file = File::open(file_path)?;
        let reader = io::BufReader::new(file);

        // Read lines and parse country scores
        let mut countries: Vec<(String, f64)> = reader
            .lines()
            .filter_map(|line| line.ok()) // Ensure valid lines
            .filter_map(|line| {
                let mut parts = line.split(',');
                let country = parts.next()?.trim().to_string();
                let score: f64 = parts
                    .nth(0)? // Select the first column for "WomenPeaceAndSecurityIndex_Score_2023"
                    .trim()
                    .parse()
                    .ok()?;
                Some((country, score))
            })
            .collect();

        // Sort countries by score in ascending order 
        countries.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        // Find the safest country
        let most_dangerous_country = countries
            .first()
            .map(|(country, _)| country.clone().trim_matches('"').to_string()) // Remove extra quotes
            .unwrap_or_else(|| "Unknown".to_string());

        // Find the most dangerous country
        let safest_country = countries
            .last()
            .map(|(country, _)| country.clone().trim_matches('"').to_string()) // Remove extra quotes
            .unwrap_or_else(|| "Unknown".to_string());

        // Assert that the safest country and most dangerous country match expected values
        assert_eq!(most_dangerous_country, "Afghanistan");
        assert_eq!(safest_country, "Denmark");

        Ok(())
    }

    #[test]
    fn test_create_similarity_matrix() {
        // Library imports
        use ndarray::array;
        use approx::assert_abs_diff_eq;

        let data = vec![
            array![1.0, 2.0],
            array![4.0, 6.0],
            array![7.0, 1.0],
        ];

        // Generates similarity matrix based on my function
        let similarity_matrix = create_similarity_matrix(&data);

        // Calculated expected matrix for this example
        let expected = array![
            [0.0, 5.0, 6.08276253],
            [5.0, 0.0, 5.83095189],
            [6.08276253, 5.83095189, 0.0],
        ];

        // Verifies the dimensions of the similarity matrix
        assert_eq!(
            similarity_matrix.shape(),
            expected.shape(),
            "Matrix shape mismatch: expected {:?}, got {:?}",
            expected.shape(),
            similarity_matrix.shape()
        );
        
        // Checks that each value is equal to eachother
        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(
                    similarity_matrix[[i, j]],
                    expected[[i, j]],
                    epsilon = 1e-6
                );
            }
        }


        // Lastly confirms similarity matrix is symmetric
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(
                    similarity_matrix[[i, j]],
                    similarity_matrix[[j, i]],
                    "Matrix is not symmetric at ({}, {})",
                    i,
                    j
                );
            }
        }
    }
}
