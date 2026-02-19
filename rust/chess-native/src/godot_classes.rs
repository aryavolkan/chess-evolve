use godot::prelude::*;

use crate::board::ChessBoard;
use crate::nn::DenseNetwork;

#[derive(GodotClass)]
#[class(base=RefCounted)]
pub struct RustChessBoard {
    board: ChessBoard,
    base: Base<RefCounted>,
}

#[godot_api]
impl IRefCounted for RustChessBoard {
    fn init(base: Base<RefCounted>) -> Self {
        Self {
            board: ChessBoard::startpos(),
            base,
        }
    }
}

#[godot_api]
impl RustChessBoard {
    #[func]
    pub fn from_fen(&mut self, _fen: GString) {
        self.board = ChessBoard::startpos();
    }

    #[func]
    pub fn get_legal_moves(&self) -> PackedInt32Array {
        let moves = self.board.get_legal_moves();
        let mut arr = PackedInt32Array::new();
        for mv in moves {
            arr.push(mv as i32);
        }
        arr
    }

    #[func]
    pub fn make_move(&mut self, mv: i32) {
        self.board = self.board.make_move(mv as u32);
    }

    #[func]
    pub fn is_in_check(&self) -> bool {
        self.board.is_in_check(self.board.side_to_move)
    }

    #[func]
    pub fn side_to_move(&self) -> i32 {
        self.board.side_to_move as i32
    }
}

#[derive(GodotClass)]
#[class(base=RefCounted)]
pub struct RustDenseNetwork {
    net: DenseNetwork,
    base: Base<RefCounted>,
}

#[godot_api]
impl IRefCounted for RustDenseNetwork {
    fn init(base: Base<RefCounted>) -> Self {
        Self {
            net: DenseNetwork::from_weights(
                0,
                0,
                0,
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
            ),
            base,
        }
    }
}

#[godot_api]
impl RustDenseNetwork {
    #[func]
    pub fn setup(&mut self, input_size: i32, hidden_size: i32, output_size: i32) {
        let input_size = input_size as usize;
        let hidden_size = hidden_size as usize;
        let output_size = output_size as usize;
        self.net = DenseNetwork::from_weights(
            input_size,
            hidden_size,
            output_size,
            vec![0.0; input_size * hidden_size],
            vec![0.0; hidden_size],
            vec![0.0; hidden_size * output_size],
            vec![0.0; output_size],
        );
    }

    #[func]
    pub fn set_weights(
        &mut self,
        weights_ih: PackedFloat32Array,
        biases_h: PackedFloat32Array,
        weights_ho: PackedFloat32Array,
        biases_o: PackedFloat32Array,
    ) {
        self.net.weights_ih = weights_ih.to_vec();
        self.net.biases_h = biases_h.to_vec();
        self.net.weights_ho = weights_ho.to_vec();
        self.net.biases_o = biases_o.to_vec();
    }

    #[func]
    pub fn forward(&self, inputs: PackedFloat32Array) -> PackedFloat32Array {
        let input_vec: Vec<f32> = inputs.to_vec();
        let out = self.net.forward(&input_vec);
        let mut arr = PackedFloat32Array::new();
        for v in out {
            arr.push(v);
        }
        arr
    }
}
