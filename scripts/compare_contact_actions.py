#!/usr/bin/env python3
"""
Compare action xyz magnitudes between contact and non-contact labeled steps in an HDF5 file.
"""

import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Compare action norms: contact vs non-contact")
    parser.add_argument("hdf5_path", type=str, help="Path to the HDF5 file")
    parser.add_argument("--max_norm", type=float, default=None, help="Clip x-axis to this maximum norm value")
    parser.add_argument(
        "--mode",
        choices=["xyz", "pose"],
        default="xyz",
        help="'xyz': norm of xyz displacement only (default); 'pose': norm of [Δxyz, Δquat] concatenated",
    )
    args = parser.parse_args()

    contact_norms = []
    non_contact_norms = []

    with h5py.File(args.hdf5_path, "r") as f:
        demos = sorted([k for k in f["data"].keys() if k.startswith("demo_")])
        print(f"Found {len(demos)} episodes")

        for demo_key in demos:
            ep = f[f"data/{demo_key}"]
            actions = np.array(ep["actions"])
            is_contact = np.array(ep["obs/is_contact"]).flatten()

            if args.mode == "xyz":
                xyz = actions[:, :3]
                deltas = np.diff(xyz, axis=0)
            else:
                # pose mode: [x, y, z, qx, qy, qz, qw]
                pose = actions[:, :7]
                # Flip quaternion sign so consecutive quats take the shorter arc
                quats = pose[:, 3:]
                for i in range(1, len(quats)):
                    if np.dot(quats[i], quats[i - 1]) < 0:
                        quats[i] = -quats[i]
                deltas = np.diff(pose, axis=0)
            norms = np.linalg.norm(deltas, axis=1)
            # is_contact aligns with actions; drop the first step since diff loses one
            is_contact = is_contact[1:]

            contact_mask = is_contact == 1
            contact_norms.extend(norms[contact_mask].tolist())
            non_contact_norms.extend(norms[~contact_mask].tolist())

    contact_norms = np.array(contact_norms)
    non_contact_norms = np.array(non_contact_norms)

    print(f"Contact steps:     {len(contact_norms):6d}  mean norm = {contact_norms.mean():.6f}")
    print(f"Non-contact steps: {len(non_contact_norms):6d}  mean norm = {non_contact_norms.mean():.6f}")

    # Shared bins for both histograms
    all_norms = np.concatenate([contact_norms, non_contact_norms])
    x_max = args.max_norm if args.max_norm is not None else all_norms.max()
    bins = np.linspace(all_norms.min(), x_max, 200)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.set_xlim(all_norms.min(), x_max)

    xlabel = "Action XYZ norm" if args.mode == "xyz" else "Action pose norm [Δxyz, Δquat]"
    suptitle = (
        "Action XYZ Norm Distribution: Contact vs Non-contact"
        if args.mode == "xyz"
        else "Action Pose Norm Distribution: Contact vs Non-contact"
    )

    ax1.hist(contact_norms, bins=bins, color="tab:red", alpha=0.8)
    ax1.set_title(f"Contact (n={len(contact_norms)}, mean={contact_norms.mean():.4f})")
    ax1.set_ylabel("Count")
    ax1.set_xlabel(xlabel)

    ax2.hist(non_contact_norms, bins=bins, color="tab:blue", alpha=0.8)
    ax2.set_title(f"Non-contact (n={len(non_contact_norms)}, mean={non_contact_norms.mean():.4f})")
    ax2.set_ylabel("Count")
    ax2.set_xlabel(xlabel)

    fig.suptitle(suptitle)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
