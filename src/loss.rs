use ndarray::{Array2, array};
use crate::FType;

pub fn binary_cross_entropy(preds: &Array2<FType>, labels: &Array2<FType>, reduction: &str) -> Array2<FType> {
    let m = preds.nrows() as FType;
    let loss: Array2<FType> = -1./m * (labels.t().dot(&preds.mapv(FType::ln)) + (1. - labels).t().dot(&(1. - preds).mapv(FType::ln)));
    match reduction {
        "mean" => array![[loss.mean().unwrap()]],
        "sum" => array![[loss.sum()]],
        _ => loss,
    }
}