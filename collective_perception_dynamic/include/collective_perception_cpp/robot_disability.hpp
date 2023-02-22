#ifndef COLLECTIVE_PERCEPTION_ROBOT_DISABLILITY_HPP
#define COLLECTIVE_PERCEPTION_ROBOT_DISABLILITY_HPP

#include <string>

#include "argos3/core/utility/configuration/argos_exception.h"

// Define Buzz keyword macros; if this is modified then `body.bzz` will have to be modified as well
#define BUZZ_KEYWORD_MOTION_DIS "motion_disabled"
#define BUZZ_KEYWORD_COMMS_DIS "comms_disabled"
#define BUZZ_KEYWORD_SENSE_DIS "sense_disabled"

using namespace argos;

/**
 * @brief Enumeration class for disability type
 *
 */
enum class DisabilityType
{
    motion = 0,
    comms,
    sense
};

/**
 * @brief Enumeration class for disability status of the entire swarm
 *
 */
enum class SwarmDisabilityStatus
{
    inactive = 0,
    executing,
    active
};

/**
 * @brief Get the Buzz keyword for a specific disability
 *
 * @param dt Enum for the disability type
 * @return std::string Buzz keyword corresponding to the provided enum
 */
inline std::string GetBuzzDisabilityKeyword(const DisabilityType &dt)
{
    switch (dt)
    {
    case DisabilityType::motion:
    {
        return BUZZ_KEYWORD_MOTION_DIS;
    }
    case DisabilityType::comms:
    {
        return BUZZ_KEYWORD_COMMS_DIS;
    }
    case DisabilityType::sense:
    {
        return BUZZ_KEYWORD_SENSE_DIS;
    }
    default:
    {
        THROW_ARGOSEXCEPTION("Invalid disability type specified as Buzz keyword!");
    }
    }
}

#endif