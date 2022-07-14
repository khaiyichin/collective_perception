import setuptools
import setuptools.command.install
import subprocess
import os

class GenerateProtolPb2Command(setuptools.command.install.install):
    description = "Generate *pb2.py files from .proto schema that allows relative import."

    def run(self):

        # Define paths to protobuf schema
        proto_path = os.path.abspath("../collective_perception_dynamic/proto")
        util_proto_path = os.path.join(proto_path, "util.proto")
        simulation_set_proto_path = os.path.join(proto_path, "simulation_set.proto")
        pb2_relative_dir_path = "pb2"

        # Defind command to generate vanilla pb2 files
        protoc_cmd = "protoc"
        protoc_args = "--python_out={0} --proto_path={1} {2} {3}" \
            .format(pb2_relative_dir_path, proto_path, util_proto_path, simulation_set_proto_path).split(" ")
        # generate_command = protoc_cmd + " " + protoc_args

        # Define command to generate protoletariat converted pb2 files
        protol_cmd = "protol"
        protol_args = [
            # "--create-package",
            "--in-place",
            "--python-out",
            pb2_relative_dir_path,
            protoc_cmd,
            "--proto-path",
            proto_path,
            util_proto_path,
            simulation_set_proto_path
        ]

        super().run()

        # Execute commands
        subprocess.run([protoc_cmd, *protoc_args])
        subprocess.run([protol_cmd, *protol_args])
        

setuptools.setup(
    name="collective_perception_py",
    version="0.2.0",
    description= "Python simulator for studying collective perception problems with static communication topologies.",
    author="Khai Yi Chin",
    author_email="khaiyichin@gmail.com",
    url="https://github.com/khaiyichin/collective_perception",
    packages=setuptools.find_packages(),
    cmdclass={
        "install": GenerateProtolPb2Command
    },
    scripts=[
        "scripts/visualize_multi_agent_data_static.py",
        "scripts/visualize_multi_agent_data_dynamic.py",
        "scripts/serial_data_info.py",
        "scripts/convert_exp_data_to_viz_data_group.py",
        "scripts/convert_sim_stats_set_to_viz_data_group.py",
        "scripts/multi_agent_sim_static.py"
    ]
)