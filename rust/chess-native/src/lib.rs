use godot::prelude::*;

mod board;
mod nn;
mod godot_classes;

struct ChessNativeExtension;

#[gdextension]
unsafe impl ExtensionLibrary for ChessNativeExtension {}
