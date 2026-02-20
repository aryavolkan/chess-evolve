#![allow(clippy::needless_range_loop, dead_code)]

pub struct DenseNetwork {
    pub input_size: usize,
    pub hidden_size: usize,
    pub output_size: usize,
    pub weights_ih: Vec<f32>,
    pub biases_h: Vec<f32>,
    pub weights_ho: Vec<f32>,
    pub biases_o: Vec<f32>,
}

impl DenseNetwork {
    pub fn from_weights(
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
        weights_ih: Vec<f32>,
        biases_h: Vec<f32>,
        weights_ho: Vec<f32>,
        biases_o: Vec<f32>,
    ) -> Self {
        Self {
            input_size,
            hidden_size,
            output_size,
            weights_ih,
            biases_h,
            weights_ho,
            biases_o,
        }
    }

    #[allow(clippy::needless_range_loop)]
    pub fn forward_into(&self, inputs: &[f32], hidden: &mut [f32], output: &mut [f32]) {
        debug_assert_eq!(hidden.len(), self.hidden_size);
        debug_assert_eq!(output.len(), self.output_size);

        for h in 0..self.hidden_size {
            let mut sum = self.biases_h[h];
            let offset = h * self.input_size;
            for i in 0..self.input_size {
                sum += self.weights_ih[offset + i] * inputs[i];
            }
            hidden[h] = sum.tanh();
        }

        for o in 0..self.output_size {
            let mut sum = self.biases_o[o];
            let offset = o * self.hidden_size;
            for h in 0..self.hidden_size {
                sum += self.weights_ho[offset + h] * hidden[h];
            }
            output[o] = sum.tanh();
        }
    }

    pub fn forward(&self, inputs: &[f32]) -> Vec<f32> {
        let mut hidden = vec![0.0f32; self.hidden_size];
        let mut output = vec![0.0f32; self.output_size];
        self.forward_into(inputs, &mut hidden, &mut output);
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn softmax(values: &[f32]) -> Vec<f32> {
        let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut exp_vals: Vec<f32> = values.iter().map(|v| (v - max).exp()).collect();
        let sum: f32 = exp_vals.iter().sum();
        for v in &mut exp_vals {
            *v /= sum;
        }
        exp_vals
    }

    #[test]
    fn forward_pass_matches_expected() {
        let net = DenseNetwork::from_weights(
            2,
            2,
            2,
            vec![1.0, 0.0, 0.0, 1.0],
            vec![0.0, 0.0],
            vec![1.0, 0.0, 0.0, 1.0],
            vec![0.0, 0.0],
        );
        let inputs = [0.5, -0.5];
        let output = net.forward(&inputs);
        assert_eq!(output.len(), 2);
        let expected0 = inputs[0].tanh().tanh();
        let expected1 = inputs[1].tanh().tanh();
        let eps = 1e-6;
        assert!((output[0] - expected0).abs() < eps);
        assert!((output[1] - expected1).abs() < eps);
    }

    #[test]
    fn softmax_sums_to_one() {
        let net = DenseNetwork::from_weights(
            3,
            2,
            3,
            vec![0.1, 0.2, 0.3, -0.1, 0.0, 0.2],
            vec![0.0, 0.1],
            vec![0.2, -0.1, 0.0, 0.1, 0.3, -0.2],
            vec![0.0, 0.0, 0.1],
        );
        let output = net.forward(&[0.25, -0.5, 0.75]);
        let probs = softmax(&output);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(probs.iter().all(|p| *p > 0.0));
    }
}
