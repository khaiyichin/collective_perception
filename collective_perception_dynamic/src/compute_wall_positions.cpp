#include <iostream>
#include <string>
#include <cmath>
#include <stdio.h>
#include <tuple>
#include <array>
#include <unistd.h> // getopt()
#include <libgen.h> // basename()

#define DEFAULT_THICKNESS 0.1
#define DEFAULT_HEIGHT 0.5

bool GenerateHelpMessage(const int &argc, char **argv)
{
    int opt;

    while ((opt = getopt(argc, argv, "h")) != -1)
    {
        switch (opt)
        {
        case 'h':
        {
            printf("usage: %s [-h]\n", basename(argv[0]));
            printf("\nComputes and generates the wall positions required for a specified swarm density.\n");

            printf("\noptions:\n");
            printf("\t-h\t\tshow this help message and exit\n");

            return true;
        }

        case '?': // unknown character
        {
            throw std::runtime_error("Unknown argument!");
        }
        }
    }
    return false;
}

double ComputeGenericWallPosition(const int &robots,
                                  const double &density,
                                  const double &comms_rad,
                                  const double &thickness)
{

    double comms_area = robots * M_PI * std::pow(comms_rad, 2);
    double walkable_length = std::sqrt(comms_area / density);

    return (walkable_length + thickness) / 2;
}

std::string GenerateArgosXMLOutput(const int &robots,
                                   const double &comms_rad,
                                   const double &thickness,
                                   const double &height,
                                   const double &pos)
{
    // Define an alias container (wall_north, (size_x, size_y, size_z), (pos_x, pos_y))
    using tuple_id_size_pos = std::tuple<std::string, std::array<double, 3>, std::array<double, 3>>;

    // Compute the size of the wall
    double wall_length = std::ceil(pos * 2);

    // Assign the size and coordinates for each wall
    tuple_id_size_pos w_north = tuple_id_size_pos{"wall_north",
                                                  {wall_length, thickness, height},
                                                  {0, pos, 0}};
    tuple_id_size_pos w_south = tuple_id_size_pos{"wall_south",
                                                  {wall_length, thickness, height},
                                                  {0, -pos, 0}};
    tuple_id_size_pos w_east = tuple_id_size_pos{"wall_east",
                                                 {thickness, wall_length, height},
                                                 {pos, 0, 0}};
    tuple_id_size_pos w_west = tuple_id_size_pos{"wall_west",
                                                 {thickness, wall_length, height},
                                                 {-pos, 0, 0}};

    std::array<tuple_id_size_pos, 4> walls = {w_north, w_south, w_east, w_west};

    // Create XML output for the box nodes
    std::string output = "";

    for (auto &wall : walls)
    {
        std::string id = std::get<0>(wall);

        auto wall_size = std::get<1>(wall);
        std::string size_str = std::to_string(wall_size[0]) + ", " + std::to_string(wall_size[1]) + ", " + std::to_string(wall_size[2]);

        auto wall_pos = std::get<2>(wall);
        std::string pos_str = std::to_string(wall_pos[0]) + ", " + std::to_string(wall_pos[1]) + ", " + std::to_string(wall_pos[2]);

        std::string box_str =
            "\t<box id=\"" + id + "\" size=\"" + size_str + "\" movable=\"false\">" +
            "\n\t    <body position=\"" + pos_str + "\" orientation=\"0, 0, 0\" />" +
            "\n\t</box>\n";

        output += box_str;
    }

    // Create XML output for the distribute nodes
    std::string dist_pos_min_str = std::to_string(-pos) + ", " + std::to_string(-pos) + ", 0";
    std::string dist_pos_max_str = std::to_string(pos) + ", " + std::to_string(pos) + ", 0";

    std::string distribute_str =
        "\n\t<distribute> "
        "\n\t    <position method=\"uniform\" min=\"" +
        dist_pos_min_str + "\" max=\"" + dist_pos_max_str + "\" />" +
        "\n\t    <orientation method=\"uniform\" min=\"0, 0, 0\" max=\"0, 0, 0\" />" +
        "\n\t    <entity quantity=\"" + std::to_string(robots) + "\" max_trials=\"100\" base_num=\"0\">" +
        "\n\t        <kheperaiv id=\"kiv\" rab_data_size=\"50\" rab_range=\"" + std::to_string(comms_rad) + "\">" +
        "\n\t            <controller config=\"bck\" />" +
        "\n\t        </kheperaiv>" +
        "\n\t    </entity>" +
        "\n\t</distribute>";

    output += distribute_str;

    return output;
}

int main(int argc, char **argv)
{
    if (GenerateHelpMessage(argc, argv))
    {
        return 0;
    }

    // Collect user inputs
    std::string density_str, robots_str, comms_rad_str, thickness_str, height_str;

    std::cout << std::endl
              << "Please specify the desired number of robots: ";
    std::cin >> robots_str;
    int robots = std::stoi(robots_str);

    std::cout << "Please specify the desired swarm density: ";
    std::cin >> density_str;
    double density = std::stod(density_str);

    std::cout << "Please specify the radius of the communication range of the robot in m: ";
    std::cin >> comms_rad_str;
    double comms_rad = std::stod(comms_rad_str);

    std::cin.ignore(); // clear newline character from the buffer to allow std::getline to work properly

    std::cout << "Please specify the desired wall thickness in m (optional): ";
    std::getline(std::cin, thickness_str);
    double thickness = thickness_str.empty() ? DEFAULT_THICKNESS : std::stod(thickness_str);

    std::cout << "Please specify the desired wall height in m (optional): ";
    std::getline(std::cin, height_str);
    double height = height_str.empty() ? DEFAULT_HEIGHT : std::stod(height_str);

    std::cout << std::endl
              << std::string(100, '*') << std::endl;

    // Compute wall length with thickness considered
    double gen_wall_pos = ComputeGenericWallPosition(robots, density, comms_rad, thickness);

    std::cout << std::endl;
    std::cout << "Based on the following specified inputs:" << std::endl;
    std::cout << "\tnumber of robots\t\t= " << std::to_string(robots) << std::endl;
    std::cout << "\tswarm density\t\t\t= " << std::to_string(density) << std::endl;
    std::cout << "\tcommunication range radius\t= " << std::to_string(comms_rad) << " m" << std::endl;
    std::cout << "\twall thickness\t\t\t= " << std::to_string(thickness) << " m" << std::endl;
    std::cout << "the corresponding wall coordinate is " << gen_wall_pos << " m." << std::endl;

    // Output XML output
    std::cout
        << std::endl
        << "Configuration for the .argos file (copy and paste this inside the <arena> node):\n"
        << std::endl
        << GenerateArgosXMLOutput(robots, comms_rad, thickness, height, gen_wall_pos) << std::endl;

    std::cout << std::endl
              << std::string(100, '*') << std::endl;

    return 0;
}