import launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch_ros.actions import Node

def generate_launch_description():
    img_width, img_height = 480, 270
    fps = 15
    auto_parameter = True
    enable_pointcloud = False  # Set to False if you don't need pointcloud data
    return LaunchDescription([
        # Camera nodes
        GroupAction([
            # Node(
            #     package='realsense2_camera',
            #     executable='realsense2_camera_node',
            #     name='cam1',
            #     namespace='cam1',
            #     parameters=[{
            #         'serial_no': '234322306820',
            #         'depth_module.profile': f'{img_width}x{img_height}x{fps}',
            #         'rgb_camera.profile': f'{img_width}x{img_height}x{fps}',
            #         'enable_infra': False, 
            #         'enable_infra1': False,
            #         'enable_infra2': False,
            #         'enable_gyro': False,
            #         'enable_accel': False,
            #         'rgb_camera.enable_auto_exposure': auto_parameter,
            #         'depth_module.enable_auto_exposure': auto_parameter,
            #         'rgb_camera.enable_auto_white_balance': auto_parameter,
            #         'rgb_camera.white_balance':3293.0,
            #         'rgb_camera.exposure':100,
            #         'enable_infra': False, 
            #         'align_depth.enable': True,
            #         'depth_width': img_width,
            #         'depth_height': img_height,
            #         'depth_fps': fps,
            #         'color_width': img_width,
            #         'color_height': img_height,
            #         'color_fps': fps,
            #         'pointcloud.enable': enable_pointcloud,
            #     }]
            # ),
            # Node(
            #     package='realsense2_camera',
            #     executable='realsense2_camera_node',
            #     name='cam2',
            #     namespace='cam2',
            #     parameters=[{
            #         'serial_no': '239222303414',
            #         'depth_module.profile': f'{img_width}x{img_height}x{fps}',
            #         'rgb_camera.profile': f'{img_width}x{img_height}x{fps}',
            #         'enable_infra': False, 
            #         'enable_infra1': False,
            #         'enable_infra2': False,
            #         'enable_gyro': False,
            #         'enable_accel': False,
            #         'rgb_camera.enable_auto_exposure': auto_parameter,
            #         'depth_module.enable_auto_exposure': auto_parameter,
            #         'rgb_camera.enable_auto_white_balance': auto_parameter,
            #         'rgb_camera.white_balance':3330.0,
            #         'rgb_camera.exposure':100,
            #         'enable_infra': False, 
            #         'align_depth.enable': True,
            #         'depth_width': img_width,
            #         'depth_height': img_height,
            #         'depth_fps': fps,
            #         'color_width': img_width,
            #         'color_height': img_height,
            #         'color_fps': fps,
            #         'pointcloud.enable': enable_pointcloud,
            #     }]
            # ),
            Node(
                package='realsense2_camera',
                executable='realsense2_camera_node',
                name='cam3',
                namespace='cam3',
                # realsense2_camera_node is slow on SIGINT during USB teardown;
                # shrink launch's escalation timeouts so shutdown takes seconds, not 10+s.
                sigterm_timeout='2',
                sigkill_timeout='3',
                parameters=[{
                    'serial_no': '239222303046', #lower one
                    'depth_module.profile': f'{img_width}x{img_height}x{fps}',
                    'rgb_camera.profile': f'{img_width}x{img_height}x{fps}',
                    'enable_infra': False,
                    'enable_infra1': False,
                    'enable_infra2': False,
                    'enable_gyro': False,
                    'enable_accel': False,
                    'rgb_camera.enable_auto_exposure': auto_parameter,
                    'depth_module.enable_auto_exposure': auto_parameter,
                    'rgb_camera.enable_auto_white_balance': auto_parameter,
                    'rgb_camera.white_balance':4000.0,
                    'rgb_camera.saturation':50.0,
                    'depth_module.exposure':14725,
                    'rgb_camera.exposure':130,
                    'align_depth.enable': True,
                    'depth_width': img_width,
                    'depth_height': img_height,
                    'depth_fps': fps,
                    'color_width': img_width,
                    'color_height': img_height,
                    'color_fps': fps,
                    'pointcloud.enable': enable_pointcloud,
                }]
            ),
            # Node(
            #     package='realsense2_camera',
            #     executable='realsense2_camera_node',
            #     name='tim_camera',
            #     namespace='cam4',
            #     parameters=[{
            #         'serial_no': '218722271574',
            #         'depth_module.profile': f'{img_width}x{img_height}x{fps}',
            #         'rgb_camera.profile': f'{img_width}x{img_height}x{fps}',
            #         'enable_infra': False, 
            #         'enable_infra1': False,
            #         'enable_infra2': False,
            #         'enable_gyro': False,
            #         'enable_accel': False,
            #         'depth_module.enable_auto_exposure':False,
            #         'depth_module.exposure':24557,
            #         'depth_module.brightness':0,
            #         'align_depth.enable': True,
            #         'depth_width': img_width,
            #         'depth_height': img_height,
            #         'depth_fps': fps,
            #         'color_width': img_width,
            #         'color_height': img_height,
            #         'color_fps': fps,
            #         'camera_name': 'cam4',
            #     }]
            # ), # Old
            Node(
                package='realsense2_camera',
                executable='realsense2_camera_node',
                name='cam4',
                namespace='cam4',
                sigterm_timeout='2',
                sigkill_timeout='3',
                parameters=[{
                    'serial_no': '218722271574',
                    'depth_module.profile': f'{img_width}x{img_height}x{fps}',
                    'rgb_camera.profile': f'{img_width}x{img_height}x{fps}',
                    'enable_infra': False, 
                    'enable_infra1': False,
                    'enable_infra2': False,
                    'enable_gyro': False,
                    'enable_accel': False,
                    'depth_module.enable_auto_exposure':False,
                    'depth_module.exposure':40557,
                    'depth_module.brightness':0,
                    'align_depth.enable': True,
                    'depth_width': img_width,
                    'depth_height': img_height,
                    'depth_fps': fps,
                    'color_width': img_width,
                    'color_height': img_height,
                    'color_fps': fps,
                }]
            ),
        ]),

        # Static transforms for each camera

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='bob_link_broadcaster',
            arguments=['0.61482', '0.523489', '0.638426', '0.321738', '0.30324', '-0.611973', '0.655759', 'fr3_link0', 'cam1_link']
            # arguments=['0.55482', '0.483489', '0.659426', '0.321738', '0.30324', '-0.611973', '0.655759', 'fr3_link0', 'cam1_link']
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='mel_link_broadcaster',
            arguments=['0.508074', '-0.522051', '0.682903', '-0.330997', '0.343186', '0.628994', '0.614029', 'fr3_link0', 'cam2_link']
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='dave_link_broadcaster',
            arguments=['1.260047', '-0.08', '0.625', '-0.300660', '-0.006615', '0.953625', '0.012632', 'fr3_link0', 'cam3_link']
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='workspace_link_broadcaster',
            arguments=['0.5', '0.0', '0.0', '0.0', '0.0', '0.0', '1.0', 'fr3_link0', 'workspace_link']
        ),
    ])
