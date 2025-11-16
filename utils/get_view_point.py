import open3d as o3d
import sys
import os

def main():
    """
    An interactive tool to find and print the camera parameters for a desired view.
    """
    print("--- Open3D Viewpoint Finder ---")

    # Check if a path to a point cloud file was provided
    if len(sys.argv) < 2:
        print("\nUsage: python get_view_point.py <path_to_your_point_cloud.ply>")
        print("Example: python get_view_point.py ./my_point_cloud_sequence/frame_0.ply")
        return

    pcd_path = sys.argv[1]

    if not os.path.exists(pcd_path):
        print(f"Error: File not found at '{pcd_path}'")
        return

    print(f"\nLoading point cloud: {pcd_path}")
    pcd = o3d.io.read_point_cloud(pcd_path)

    print("\n" + "="*50)
    print("INSTRUCTIONS:")
    print("1. Use your mouse to ROTATE (left-click + drag), PAN (right-click + drag), and ZOOM (scroll wheel).")
    print("2. Adjust the view until it looks exactly how you want it for the video.")
    print("3. Press 'c' on your keyboard. This will copy the camera parameters to your clipboard and print them in the terminal.")
    print("4. Press 'q' on your keyboard to close the window.")
    print("="*50 + "\n")

    # This function will be called when the user presses 'c'
    def copy_camera_params(vis):
        params = vis.get_view_control().convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters("camera_viewpoint.json", params)
        print("Camera parameters have been saved to 'camera_viewpoint.json'.")
        print("You can now close the window by pressing 'q'.")
        return False

    key_to_callback = {}
    key_to_callback[ord("C")] = copy_camera_params

    # Run the visualizer
    o3d.visualization.draw_geometries_with_key_callbacks([pcd], key_to_callback)


if __name__ == "__main__":
    main()
    # usage: python get_view_point.py ./my_point_cloud_sequence/frame_0.ply
