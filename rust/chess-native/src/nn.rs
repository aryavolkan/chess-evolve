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

    pub fn forward(&self, inputs: &[f32]) -> Vec<f32> {
        let mut hidden = vec![0.0f32; self.hidden_size];
        let mut output = vec![0.0f32; self.output_size];

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

        output
    }
}
