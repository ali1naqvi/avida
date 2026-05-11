/*
 *  cTextViewerDriver.cc
 *  Avida
 *
 *  Created by David on 12/11/05.
 *  Copyright 1999-2009 Michigan State University. All rights reserved.
 *
 *
 *  This file is part of Avida.
 *
 *  Avida is free software; you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License
 *  as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 *
 *  Avida is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License along with Avida.
 *  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "cTextViewerDriver.h"

#include "avida/core/Context.h"
#include "avida/core/World.h"

#include "cHardwareBase.h"
#include "cOrganism.h"
#include "cPopulation.h"
#include "cPopulationCell.h"
#include "cStats.h"
#include "cString.h"
#include "cView.h"
#include "cWorld.h"

#include <cstdlib>

using namespace std;


cTextViewerDriver::cTextViewerDriver(cWorld* world)
  : cTextViewerDriver_Base(world), m_pause(false), m_firstupdate(true)
{
  GlobalObjectManager::Register(this);
  world->SetDriver(this);

  m_view = new cView(world, this);
  m_view->SetViewMode(-1);    // Set the view mode to its default value.
}

cTextViewerDriver::~cTextViewerDriver()
{
  GlobalObjectManager::Unregister(this);
  
  if (m_view != NULL) EndProg(0);
}


void cTextViewerDriver::Run()
{
  cPopulation& population = m_world->GetPopulation();
  cStats& stats = m_world->GetStats();
  
  const double point_mut_prob = m_world->GetConfig().POINT_MUT_PROB.Get() +
                                m_world->GetConfig().POINT_INS_PROB.Get() +
                                m_world->GetConfig().POINT_DEL_PROB.Get() +
                                m_world->GetConfig().DIV_LGT_PROB.Get();
  
  void (cPopulation::*ActiveProcessStep)(cAvidaContext& ctx, double step_size, int cell_id) = &cPopulation::ProcessStep;
  if (m_world->GetConfig().SPECULATIVE.Get() &&
      m_world->GetConfig().THREAD_SLICING_METHOD.Get() != 1 && !m_world->GetConfig().IMPLICIT_REPRO_END.Get() && point_mut_prob == 0.0) {
    ActiveProcessStep = &cPopulation::ProcessStepSpeculative;
  }
  
  cAvidaContext& ctx = m_world->GetDefaultContext();
  Avida::Context new_ctx(this, &m_world->GetRandom());
  
  while (!m_done) {
    
    m_world->GetEvents(ctx);
    if (m_done == true) break;
    
    // Increment the Update.
    stats.IncCurrentUpdate();
    
    population.ProcessPreUpdate();

    // Handle all data collection for previous update.
    if (stats.GetUpdate() > 0) {
      // Tell the stats object to do update calculations and printing.
      stats.ProcessUpdate();
    }
    
    
    // Process the update.
    const int UD_size = m_world->CalculateUpdateSize();
    const double step_size = 1.0 / (double) UD_size;
    
    if (m_pause) {
      m_view->Pause();
      m_pause = false;
      
      // This is needed to have the top bar drawn properly; I'm not sure why...
      if (m_firstupdate) {
        m_view->Refresh(ctx);
        m_firstupdate = false;
      }
    }
    
    // Are we stepping through an organism?
    if (m_view->GetStepOrganism() != -1) {  // Yes we are!
                                            // Keep the viewer informed about the organism we are stepping through...
      for (int i = 0; i < UD_size; i++) {
        if (population.GetNumOrganisms() == 0) {
          break;
        }
        const int next_id = population.ScheduleOrganism();
        if (next_id == m_view->GetStepOrganism()) {
          m_view->NotifyUpdate(ctx);
          m_view->NewUpdate(ctx);
          
          // This is needed to have the top bar drawn properly; I'm not sure why...
          if (m_firstupdate) {
            m_view->Refresh(ctx);
            m_firstupdate = false;
          }
        }
        (population.*ActiveProcessStep)(ctx, step_size, next_id);
      }
    }
    else {
      for (int i = 0; i < UD_size; i++) {
        if (population.GetNumOrganisms() == 0) {
          break;
        }
        (population.*ActiveProcessStep)(ctx, step_size, population.ScheduleOrganism());
      }
    }
    
    
    // end of update stats...
    population.ProcessPostUpdate(ctx);
    
    m_world->ProcessPostUpdate(ctx);
    
    // Setup the viewer for the new update.
    if (m_view->GetStepOrganism() == -1) {
      m_view->NewUpdate(ctx);
 
      // This is needed to have the top bar drawn properly; I'm not sure why...
      if (m_firstupdate) {
        m_view->Refresh(ctx);
        m_firstupdate = false;
      }
    }
    
    
    // Do Point Mutations
    if (point_mut_prob > 0 ) {
      for (int i = 0; i < population.GetSize(); i++) {
        if (population.GetCell(i).IsOccupied()) {
          int num_mut = population.GetCell(i).GetOrganism()->GetHardware().PointMutate(ctx);
          population.GetCell(i).GetOrganism()->IncPointMutations(num_mut);
        }
      }
    }
    
    m_world->GetNewWorld()->PerformUpdate(new_ctx, stats.GetUpdate());
    
    // Exit conditons...
    if ((population.GetNumOrganisms() == 0) && m_world->AllowsEarlyExit()) {
      m_done = true;
    }
  }
}

void cTextViewerDriver::RaiseException(const cString& in_string)
{
  m_view->NotifyError(in_string);
}

void cTextViewerDriver::RaiseFatalException(int exit_code, const cString& in_string)
{
  m_view->NotifyError(in_string);
  exit(exit_code);
}

void cTextViewerDriver::NotifyComment(const cString& in_string)
{
  m_view->NotifyComment(in_string);
}

void cTextViewerDriver::NotifyWarning(const cString& in_string)
{
  m_view->NotifyWarning(in_string);
}
