"""Inspect a training snapshot."""
import sys, torch
snap = torch.load(sys.argv[1], map_location="cpu")
print(f"Step: {snap['step']}")
print(f"Best val loss: {snap['best_val_loss']}")
print(f"Elapsed: {snap['elapsed']/3600:.1f}h")
print(f"Epoch: {snap['epoch']}")
if snap.get("val_losses"):
    print(f"Val losses (first 3): {snap['val_losses'][:3]}")
    print(f"Val losses (last 3): {snap['val_losses'][-3:]}")
