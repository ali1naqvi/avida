//////////////////////////////////////////////////////////////////////////////
// Copyright (C) 1993 - 2001 California Institute of Technology             //
//                                                                          // 
// Read the COPYING and README files, or contact 'avida@alife.org',         //
// before continuing.  SOME RESTRICTIONS MAY APPLY TO USE OF THIS FILE.     //
//////////////////////////////////////////////////////////////////////////////

#include "cSymbolUtil.h"

#include <fstream>

#include "cHardwareBase.h"
#include "cOrganism.h"
#include "cPopulationCell.h"
#include "cViewInfo.h"

using namespace std;


char cSymbolUtil::GetBasicSymbol(const cPopulationCell & cell)
{
  if (cell.IsOccupied() == false) return ' ';
  const cOrganism & organism = *(cell.GetOrganism());
  
  Systematics::GroupPtr bg = organism.SystematicsGroup("genotype");
  Apto::SmartPtr<sGenotypeViewInfo> view_info = bg->GetData<sGenotypeViewInfo>();
  if (!view_info) {
    view_info = Apto::SmartPtr<sGenotypeViewInfo>(new sGenotypeViewInfo);
    bg->AttachData(view_info);
  }
  return view_info->symbol;
}

char cSymbolUtil::GetModifiedSymbol(const cPopulationCell & cell)
{
  if (cell.IsOccupied() == false) return ' ';
  const cOrganism & organism = *(cell.GetOrganism());
  
  const bool modifier = organism.GetPhenotype().IsModifier();
  const bool modified = organism.GetPhenotype().IsModified();
  
  // 'I' = Injector     'H' = Host (Injected into)
  // 'B' = Both         '-' = Neither
  
  if (modifier == true && modified == true)  return 'B';
  if (modifier == true) return 'I'-6;
  if (modified == true) return 'H'-6;
  return '-';
}

char cSymbolUtil::GetResourceSymbol(const cPopulationCell & cell)
{
  // @CAO Not Implemented yet...
  if (cell.IsOccupied() == false) return ' ';
  return '.';
}

char cSymbolUtil::GetAgeSymbol(const cPopulationCell & cell)
{
  if (cell.IsOccupied() == false) return ' ';
  const cOrganism & organism = *(cell.GetOrganism());
  
  const int age = organism.GetPhenotype().GetAge();
  if (age < 0) return '-';
  if (age < 10) return (char) ('0' + age);
  if (age < 20) return 'X';
  if (age < 80) return 'L';
  if (age < 200) return 'C';
  
  return '+';
}

char cSymbolUtil::GetBreedSymbol(const cPopulationCell & cell)
{
  if (cell.IsOccupied() == false) return ' ';
  const cOrganism & organism = *(cell.GetOrganism());
  
  if (organism.GetPhenotype().ParentTrue() == true) return '*';
  return '-';
}

char cSymbolUtil::GetParasiteSymbol(const cPopulationCell & cell)
{
  if (cell.IsOccupied() == false) return ' ';
  const cOrganism & organism = *(cell.GetOrganism());
  
  if (organism.GetNumParasites()) return 'A';
  return '-';
}

char cSymbolUtil::GetForagerColor(const cPopulationCell & cell)
{
  if (cell.IsOccupied() == false) return ' ';
  const int org_target = cell.GetOrganism()->GetForageTarget();
  
  if (org_target == -2 || org_target == -3) return 'A';                   //we just want to color the predators red, the symbol used will still be a 'P'
  else if (org_target == -1) return '1';              //no target = bold-white
  else return 'B' + org_target;                       //other valid targets = colors -> white
}

char cSymbolUtil::GetForagerSymbol(const cPopulationCell & cell)
{
  if (cell.IsOccupied() == false) return ' ';
  const int org_target = cell.GetOrganism()->GetForageTarget();
  
  if (org_target == -2) return 'P';
  else if (org_target == -3) return 'T';
  else if (org_target == -1) return '-';
  else if (org_target < 10) return '0' + org_target;
  // switch to lower case letters after digits
  else if (org_target >= 10 && org_target <= 35) return 'a' + org_target - 10;
  else return '!';
}

char cSymbolUtil::GetAVForagerColor(const cPopulationCell & cell)
{
  if (cell.HasAV() == false) return ' ';
  if (cell.HasPredAV()) return 'A';           //we just want to color the predators red, the symbol used will still be a 'P'

  const int org_target = cell.GetRandPreyAV()->GetForageTarget();
  if (org_target == -1) return '1';           //no target = bold-white
  else return 'B' + org_target;               //other valid targets = colors -> white
}

char cSymbolUtil::GetAVForagerSymbol(const cPopulationCell & cell)
{
  if (cell.HasAV() == false) return ' ';
  if (cell.HasPredAV()) {
    const int org_target = cell.GetRandPredAV()->GetForageTarget();
    if (org_target == -2) return 'P';
    else if (org_target == -3) return 'T';
  }

  const int org_target = cell.GetRandPreyAV()->GetForageTarget();
  if (org_target == -1) return '-';
  else if (org_target < 10) return '0' + org_target;
  // switch to lower case letters after digits
  else if (org_target >= 10 && org_target <= 35) return 'a' + org_target - 10;
  else return '!';
}

char cSymbolUtil::GetTerritoryColor(const cPopulationCell & cell)
{
  if (cell.GetCellDataTerritory() != -1) return 'A' + cell.GetCellDataTerritory();
  else return ' ';
}

char cSymbolUtil::GetTerritorySymbol(const cPopulationCell & cell)
{
  if (cell.GetCellDataTerritory() != -1) {
    if (cell.GetCellDataTerritory() >= 0 && cell.GetCellDataTerritory() <= 9) return '0' + cell.GetCellDataTerritory();
    else if (cell.GetCellDataTerritory() > 9 && cell.GetCellDataTerritory() <= 35) return 'A' + cell.GetCellDataTerritory() - 10;
    else if (cell.GetCellDataTerritory() > 35 && cell.GetCellDataTerritory() <= 61) return 'a' + cell.GetCellDataTerritory() - 36;
    else return '!';
  }
  else return ' ';
}

char cSymbolUtil::GetMarkedCellSymbol(const cPopulationCell & cell)
{
  if (cell.GetCellData() != 0) {
    if (cell.GetCellData() >= 0 && cell.GetCellData() <= 9) return '0' + cell.GetCellData();
    else if (cell.GetCellData() > 9 && cell.GetCellData() <= 35) return 'A' + cell.GetCellData() - 10;
    else if (cell.GetCellData() > 35 && cell.GetCellData() <= 61) return 'a' + cell.GetCellData() - 36;
    else return '!';
  }
  else return ' ';
}

char cSymbolUtil::GetMarkedCellColor(const cPopulationCell & cell)
{
  if (cell.GetCellData() != 0) {
    if (cell.GetCellDataForagerType() <= -2) return 'A';
    else if (cell.GetCellDataForagerType() == -1) return '1';
    else return 'B' + cell.GetCellDataForagerType();
  }
  else return ' ';
}

char cSymbolUtil::GetMutSymbol(const cPopulationCell & cell)
{
  if (cell.IsOccupied() == false) return ' ';
  const cOrganism & organism = *(cell.GetOrganism());
  
  if (organism.GetPhenotype().IsMutated() == true) return '*';
  return '-';
}

char cSymbolUtil::GetThreadSymbol(const cPopulationCell & cell)
{
  if (cell.IsOccupied() == false) return ' ';
  return '0' + cell.GetOrganism()->GetHardware().GetNumThreads();
}

char cSymbolUtil::GetLineageSymbol(const cPopulationCell & cell)
{
  if (cell.IsOccupied() == false) return ' ';
  const cOrganism & organism = *(cell.GetOrganism());
  
  return 'A' + (organism.GetLineageLabel() % 12);
}

static int EnergyBin(double e)
{
  if (e <= 0.0) return 0;
  if (e < 1.0) return 1;
  if (e < 5.0) return 2;
  if (e < 10.0) return 3;
  if (e < 20.0) return 4;
  if (e < 50.0) return 5;
  if (e < 100.0) return 6;
  if (e < 200.0) return 7;
  if (e < 500.0) return 8;
  return 9;
}

char cSymbolUtil::GetEnergySymbol(const cPopulationCell& cell)
{
  if (!cell.IsOccupied()) return ' ';
  const double e = cell.GetOrganism()->GetPhenotype().GetStoredEnergy();
  const int b = EnergyBin(e);
  return (char)('0' + b);
}

char cSymbolUtil::GetEnergyColor(const cPopulationCell& cell)
{
  if (!cell.IsOccupied()) return ' ';
  const double e = cell.GetOrganism()->GetPhenotype().GetStoredEnergy();
  const int b = EnergyBin(e);

  // Viewer color codes (see `cScreen::SetSymbolColor`):
  // - '1' => bold white
  // - 'G'..'L' => normal colors 1..6
  if (b <= 1) return '1';
  if (b == 2) return 'G';
  if (b == 3) return 'H';
  if (b == 4) return 'I';
  if (b == 5) return 'J';
  if (b == 6) return 'K';
  return 'L';
}

