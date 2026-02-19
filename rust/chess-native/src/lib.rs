use godot::prelude::*;

mod board;
mod godot_classes;
mod nn;

struct ChessNativeExtension;

#[gdextension]
unsafe impl ExtensionLibrary for ChessNativeExtension {}
