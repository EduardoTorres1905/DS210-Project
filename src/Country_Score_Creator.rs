use std::cmp::Ordering;
use std::fs::File;
use std::io::{self, BufRead};
use ndarray::Array2;
use ndarray::Array1;


// Function to compute Euclidean distance between two vectors (countries' score vectors)
pub fn euclidean_distance(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

// Create a similarity matrix using Euclidean distance (lower value means higher similarity)
pub fn create_similarity_matrix(data: &Vec<Array1<f64>>) -> Array2<f64> {
    let n = data.len();
    let mut similarity_matrix = Array2::<f64>::zeros((n, n));

    for i in 0..n {
        for j in i..n {
            let dist = euclidean_distance(&data[i], &data[j]);
            similarity_matrix[[i, j]] = dist;
            similarity_matrix[[j, i]] = dist;
        }
    }

    similarity_matrix
}

// Example of clustering (you could replace this with more advanced methods)
pub fn simple_clustering(similarity_matrix: &Array2<f64>) -> Vec<usize> {
    let n = similarity_matrix.shape()[0];
    let mut clusters = vec![0; n];

    // Simple placeholder clustering algorithm (just for illustration)
    for i in 0..n {
        if similarity_matrix[[i, 0]] < 5.0 { // Threshold for cluster assignment
            clusters[i] = 1;
        } else {
            clusters[i] = 2;
        }
    }

    clusters
}