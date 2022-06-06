#ifndef COLLECTIVE_PERCEPTION_LOOP_FUNCTIONS_HPP
#define COLLECTIVE_PERCEPTION_LOOP_FUNCTIONS_HPP

#include <vector>
#include "arena.hpp"

#include <argos3/core/simulator/loop_functions.h>
#include <argos3/core/simulator/entity/floor_entity.h>

using namespace argos;

class CCollectivePerceptionLoopFunctions : public CLoopFunctions
{
    public:

        CCollectivePerceptionLoopFunctions();
        virtual ~CCollectivePerceptionLoopFunctions() {}

        virtual void Init(TConfigurationNode& t_tree);
        // virtual void Reset() {};
        // virtual void Destroy() {};
        virtual CColor GetFloorColor(const CVector2& c_position_on_plane);
        // virtual void PreStep() {};

    private:

        Arena arena_;

        CFloorEntity* m_pcFloor_;
};

#endif