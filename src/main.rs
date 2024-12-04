// Tried to use k means but kept getting errors due to stability issues go over chat log for more 
// used linfa now
// Tried to name 3 predetermined clusters but some countries were mixed up such as afganistan(safest),brazil(moderelty safe(dangerous))

//testing out this 

//make cluster first and then name clusters based on score and accuracy
// use chatgpt to make larger dataset of random countrires and data 
use std::cmp::Ordering;
use std::fs::File;
use std::io::{self, BufRead};

mod DataFrame;
use DataFrame::{DataFrame as OtherDaraFrame};

mod Country_Score_Creator;
use Country_Score_Creator::{euclidean_distance, create_similarity_matrix, simple_clustering};

// Main function to load data, compute similarity, and apply clustering
fn main() -> io::Result<()> {
    // Define the file path
    let file_path = "Most Dangerous Countries.csv";

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
            let score: f64 = parts.next()?.trim().parse().ok()?;
            Some((country, score))
        })
        .collect();

    // Sort countries by score in descending order (most safe to least safe)
    countries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

    // Calculate thresholds
    let total_countries = countries.len();
    let top_threshold = total_countries / 4;
    let bottom_threshold = total_countries * 3 / 4;

    // Categorize countries
    let most_safe: Vec<_> = countries[..top_threshold].to_vec();
    let moderately_safe: Vec<_> = countries[top_threshold..bottom_threshold].to_vec();
    let most_dangerous: Vec<_> = countries[bottom_threshold..].to_vec();

    // Print results with swapped labels
    println!("Most Safe Countries for Women:");
    for (name, score) in &most_safe {
        println!("{:<30} {:.2}", name, score);
    }

    println!("\nMost Dangerous Countries for Women:");
    for (name, score) in &most_dangerous {
        println!("{:<30} {:.2}", name, score);
    }

    println!("\nModerately Safe Countries for Women:");
    for (name, score) in &moderately_safe {
        println!("{:<30} {:.2}", name, score);
    }

    Ok(())
}


 
//let types = vec![1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2];