use ndarray::{Array2, Array1};


// Function computes Euclidean distance between two arrays
pub fn euclidean_distance(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

// Creates similarity matricies using Euclidean distance 
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

// Places countries into groups based on their scores
pub fn simple_clustering(similarity_matrix: &Array2<f64>, scores: Vec<f64>) -> Vec<usize> {
    let n = similarity_matrix.shape()[0];
    let mut clusters = vec![0; n];

    // Compute the average score for each country
    let avg_scores: Vec<f64> = similarity_matrix
        .axis_iter(ndarray::Axis(0))
        .map(|row| row.mean().unwrap_or(0.0))
        .collect();

    // Assign clusters based on thresholds (Updated thresholds)
    for i in 0..n {
        if avg_scores[i] <= 0.6 {
            clusters[i] = 0; // Most Dangerous Countries (0.0 - 0.6)
        } else if avg_scores[i] <= 0.8 {
            clusters[i] = 1; // Moderately Dangerous Countries (0.6 - 0.8)
        } else {
            clusters[i] = 2; // Safest Countries (0.8 - 1.0)
        }
    }

    clusters
}





