//////////////////////////////////////////////////////////////////////////////
// Copyright (C) 1993 - 2001 California Institute of Technology             //
//                                                                          //
// Read the COPYING and README files, or contact 'avida@alife.org',         //
// before continuing.  SOME RESTRICTIONS MAY APPLY TO USE OF THIS FILE.     //
//////////////////////////////////////////////////////////////////////////////

#include "cViewInfo.h"

#include <fstream>

#include "avida/systematics/Arbiter.h"
#include "avida/systematics/Manager.h"

#include "cEnvironment.h"
#include "cPopulation.h"
#include "cPopulationCell.h"
#include "cOrganism.h"
#include "cResource.h"
#include "cResourceLib.h"

#include "cSymbolUtil.h"
#include "cScreen.h"


using namespace std;

const Apto::String sGenotypeViewInfo::ObjectKey("sGenotypeViewInfo");

static int GradientAmountBin(double amount, double min_amount, double max_amount)
{
  if (amount <= 0.0 || max_amount <= 0.0) return 0;
  if (max_amount <= min_amount) return 9;

  double frac = (amount - min_amount) / (max_amount - min_amount);
  if (frac < 0.0) frac = 0.0;
  if (frac > 1.0) frac = 1.0;

  int bin = 1 + static_cast<int>(frac * 8.999);
  if (bin < 1) bin = 1;
  if (bin > 9) bin = 9;
  return bin;
}

static char GradientAmountSymbol(double amount, double min_amount, double max_amount)
{
  return static_cast<char>('0' + GradientAmountBin(amount, min_amount, max_amount));
}

static char GradientAmountColor(double amount, double min_amount, double max_amount)
{
  const int bin = GradientAmountBin(amount, min_amount, max_amount);

  // Match the energy view's color buckets.
  if (bin <= 1) return '1';
  if (bin == 2) return 'G';
  if (bin == 3) return 'H';
  if (bin == 4) return 'I';
  if (bin == 5) return 'J';
  if (bin == 6) return 'K';
  return 'L';
}

cViewInfo::cViewInfo(cWorld* world, cView_Base* view)
: m_world(world)
, m_view(view)
{
  active_cell = NULL;
  pause_level = PAUSE_OFF;
  saved_inst_set = NULL;
  thread_lock = -1;
  step_organism_id = -1;
  map_mode=0;

  // Handle genotype managing...

  for (int i = 0; i < NUM_SYMBOLS; i++) {
    genotype_chart[i] = Systematics::GroupPtr(NULL);
    symbol_chart[i] = (char) (i + 'A');
  }
}

void cViewInfo::AddGenChart(Systematics::GroupPtr in_gen)
{
  for (int i = 0; i < NUM_SYMBOLS; i++) {
    if (genotype_chart[i] == Systematics::GroupPtr(NULL)) {
      genotype_chart[i] = in_gen;
      getViewInfo(in_gen)->symbol = symbol_chart[i];
      break;
    }
  }
}


void cViewInfo::SetupSymbolMaps(int map_mode, bool use_color)
{
  typedef char (*SymbolMethod)(const cPopulationCell & cell);
  SymbolMethod map_method = NULL;
  SymbolMethod color_method = NULL;

  switch (map_mode) {
    case MAP_BASIC:
      if (use_color) color_method = &cSymbolUtil::GetBasicSymbol;
      else map_method = &cSymbolUtil::GetBasicSymbol;
      break;
    case MAP_INJECT:
      if (use_color) color_method = &cSymbolUtil::GetModifiedSymbol;
      else map_method = &cSymbolUtil::GetModifiedSymbol;
      break;
    case MAP_RESOURCE:
      map_method = &cSymbolUtil::GetResourceSymbol;
      break;
    case MAP_AGE:
      map_method = &cSymbolUtil::GetAgeSymbol;
      break;
    case MAP_BREED_TRUE:
      if (use_color) color_method = &cSymbolUtil::GetBreedSymbol;
      else map_method = &cSymbolUtil::GetBreedSymbol;
      break;
    case MAP_PARASITE:
      if (use_color) color_method = &cSymbolUtil::GetParasiteSymbol;
      else map_method = &cSymbolUtil::GetParasiteSymbol;
      break;
    case MAP_FORAGER:
      if (use_color) color_method = &cSymbolUtil::GetForagerColor;
      map_method = &cSymbolUtil::GetForagerSymbol;
      break;
    case MAP_AVATAR:
      if (use_color) color_method = &cSymbolUtil::GetAVForagerColor;
      map_method = &cSymbolUtil::GetAVForagerSymbol;
      break;
    case MAP_TERRITORIES:
      if (m_world->GetConfig().USE_FORM_GROUPS.Get() != 0) {
        if (use_color) color_method = &cSymbolUtil::GetTerritoryColor;
        map_method = &cSymbolUtil::GetTerritorySymbol;
      }
      else {
        if (use_color) color_method = &cSymbolUtil::GetMarkedCellColor;
        map_method = &cSymbolUtil::GetMarkedCellSymbol;        
      }
      break;
    case MAP_MUTATIONS:
      if (use_color) color_method = &cSymbolUtil::GetMutSymbol;
      else map_method = &cSymbolUtil::GetMutSymbol;
      break;
    case MAP_THREAD:
      //if (use_color) color_method = &cSymbolUtil::GetThreadSymbol;
      if (use_color) color_method = &cSymbolUtil::GetThreadSymbol;
      map_method = &cSymbolUtil::GetThreadSymbol;
      break;
    case MAP_LINEAGE:
      if (use_color) color_method = &cSymbolUtil::GetLineageSymbol;
      else map_method = &cSymbolUtil::GetLineageSymbol;
      break;
    case MAP_ENERGY:
      if (use_color) color_method = &cSymbolUtil::GetEnergyColor;
      map_method = &cSymbolUtil::GetEnergySymbol;
      break;
  }

  const int num_cells = m_world->GetPopulation().GetSize();
  map.Resize(num_cells);
  color_map.Resize(num_cells);

  if (map_mode == MAP_RESOURCE) {
    cPopulation& pop = m_world->GetPopulation();
    cAvidaContext& ctx = m_world->GetDefaultContext();
    const cResourceLib& resource_lib = m_world->GetEnvironment().GetResourceLib();
    Apto::Array<double> cell_amounts(num_cells);
    double min_amount = -1.0;
    double max_amount = 0.0;

    pop.GetResourceCount().GetResources(ctx);
    for (int i = 0; i < num_cells; i++) {
      const int env_cell_id = pop.MapPopCellToEnvCellByGrid(i);
      const Apto::Array<double>& res_count = pop.GetResourceCount().GetFrozenResources(ctx, env_cell_id);
      double amount = 0.0;
      for (int r = 0; r < res_count.GetSize() && r < resource_lib.GetSize(); r++) {
        cResource* res = resource_lib.GetResource(r);
        if (res == NULL || res->GetDemeResource()) continue;
        const int habitat = res->GetHabitat();
        if (habitat == 1 || habitat == 2) continue;
        if (res_count[r] > amount) amount = res_count[r];
      }
      cell_amounts[i] = amount;
      if (amount > 0.0 && (min_amount < 0.0 || amount < min_amount)) min_amount = amount;
      if (amount > max_amount) max_amount = amount;
    }
    if (min_amount < 0.0) min_amount = 0.0;

    for (int i = 0; i < num_cells; i++) {
      map[i] = GradientAmountSymbol(cell_amounts[i], min_amount, max_amount);
      color_map[i] = GradientAmountColor(cell_amounts[i], min_amount, max_amount);
    }
    return;
  }

  for (int i = 0; i < num_cells; i++) {
    if (map_mode == 4) m_world->GetPopulation().GetCell(i).UpdateCellDataExpired();
    if (map_method == 0) map[i] = 0;
    else map[i] = (*map_method)(m_world->GetPopulation().GetCell(i));
    
    if (color_method == 0) color_map[i] = 0;
    else color_map[i] = (*color_method)(m_world->GetPopulation().GetCell(i));
  }

}


void cViewInfo::UpdateSymbols()
{
  // First, clean up the genotype_chart.
  Systematics::ManagerPtr classmgr = Systematics::Manager::Of(m_world->GetNewWorld());

  int i, pos;
  for (i = 0; i < NUM_SYMBOLS; i++) {
    if (genotype_chart[i]) {
      Systematics::Arbiter::IteratorPtr it = classmgr->ArbiterForRole("genotype")->Begin();
      pos = -1;
      int rank = 0;
      while ((it->Next()) && i < NUM_SYMBOLS) {
        if (genotype_chart[i] == it->Get()) {
          pos = rank;
          break;
        }
        rank++;
      }

      if (pos < 0) genotype_chart[i] = Systematics::GroupPtr(NULL);
      if (pos >= NUM_SYMBOLS) {
        if (Apto::StrAs(genotype_chart[i]->Properties().Get("threshold").StringValue()))
          getViewInfo(genotype_chart[i])->symbol = '+';
        else getViewInfo(genotype_chart[i])->symbol = '.';
        genotype_chart[i] = Systematics::GroupPtr(NULL);
      }
    }
  }

  // Now, fill in any missing spaces...

  Systematics::Arbiter::IteratorPtr it = classmgr->ArbiterForRole("genotype")->Begin();
  Systematics::GroupPtr bg = it->Next();
  for (int i = 0; bg && i < SYMBOL_THRESHOLD; bg = it->Next(), i++) {
    if (!InGenChart(bg)) AddGenChart(bg);
  }
}


void cViewInfo::EngageStepMode()
{
  // Steps can only be taken through the execution of a cpu when avida is
  // paused, and focued on an active cpu.
  if ( pause_level == PAUSE_ON  &&  active_cell != NULL ) {
    pause_level = PAUSE_ADVANCE_STEP;
    SetStepOrganism( active_cell->GetID() );
  }
}

void cViewInfo::DisEngageStepMode()
{
  SetStepOrganism(-1);
}

Systematics::GroupPtr cViewInfo::GetActiveGenotype()
{
  if (active_cell != NULL && active_cell->IsOccupied()) {
    return active_cell->GetOrganism()->SystematicsGroup("genotype");
  }

  return Systematics::GroupPtr(NULL);
}


cString cViewInfo::GetActiveName()
{
  if (GetActiveGenotype() == NULL) return cString("");
  return (const char*)GetActiveGenotype()->Properties().Get("name").StringValue();
}

int cViewInfo::GetActiveID()
{
  if (active_cell) return active_cell->GetID();
  return -1;
}

int cViewInfo::GetActiveGenotypeID()
{
  return GetActiveGenotype() ? GetActiveGenotype()->ID() : -1;
}

Apto::SmartPtr<sGenotypeViewInfo> cViewInfo::getViewInfo(Systematics::GroupPtr bg)
{
  Apto::SmartPtr<sGenotypeViewInfo> view_info = bg->GetData<sGenotypeViewInfo>();
  if (!view_info) {
    view_info = Apto::SmartPtr<sGenotypeViewInfo>(new sGenotypeViewInfo);
    bg->AttachData(view_info);
  }
  return view_info;
}


bool sGenotypeViewInfo::Serialize(ArchivePtr) const
{
  // @TODO - map color serialize
  assert(false);
  return false;
}
