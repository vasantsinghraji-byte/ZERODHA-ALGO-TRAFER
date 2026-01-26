# -*- coding: utf-8 -*-
"""
Portfolio Risk Management Module - Protect the Whole Portfolio!
================================================================
Manage risk at the portfolio level with correlation checks,
sector exposure limits, and concentration controls.

Example:
    >>> from core.risk import PortfolioRiskManager, PortfolioRiskConfig
    >>>
    >>> config = PortfolioRiskConfig(
    ...     max_correlation=0.7,
    ...     max_sector_exposure=30.0,
    ...     max_single_position=15.0
    ... )
    >>> risk_mgr = PortfolioRiskManager(config, price_data)
    >>>
    >>> # Check before adding position
    >>> can_add, report = risk_mgr.check_position("TATASTEEL", quantity=100)
    >>> if can_add:
    ...     execute_trade()
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Set

import numpy as np
import pandas as pd

from .correlation import CorrelationAnalyzer, CorrelationMatrix, CorrelationPair

logger = logging.getLogger(__name__)


# =============================================================================
# SECTOR DEFINITIONS
# =============================================================================

class Sector(Enum):
    """Indian market sector classifications."""
    BANKING = "Banking"
    IT = "IT"
    PHARMA = "Pharma"
    AUTO = "Auto"
    FMCG = "FMCG"
    METALS = "Metals"
    ENERGY = "Energy"
    INFRASTRUCTURE = "Infrastructure"
    TELECOM = "Telecom"
    REALTY = "Realty"
    MEDIA = "Media"
    CEMENT = "Cement"
    CHEMICALS = "Chemicals"
    CONSUMER_DURABLES = "Consumer Durables"
    FINANCIAL_SERVICES = "Financial Services"
    HEALTHCARE = "Healthcare"
    INSURANCE = "Insurance"
    UNKNOWN = "Unknown"


# Default sector mappings for major NSE stocks
DEFAULT_SECTOR_MAP: Dict[str, Sector] = {
    # Banking
    "HDFCBANK": Sector.BANKING,
    "ICICIBANK": Sector.BANKING,
    "SBIN": Sector.BANKING,
    "KOTAKBANK": Sector.BANKING,
    "AXISBANK": Sector.BANKING,
    "INDUSINDBK": Sector.BANKING,
    "BANKBARODA": Sector.BANKING,
    "PNB": Sector.BANKING,
    "FEDERALBNK": Sector.BANKING,
    "BANDHANBNK": Sector.BANKING,

    # IT
    "TCS": Sector.IT,
    "INFY": Sector.IT,
    "WIPRO": Sector.IT,
    "HCLTECH": Sector.IT,
    "TECHM": Sector.IT,
    "LTIM": Sector.IT,
    "MPHASIS": Sector.IT,
    "COFORGE": Sector.IT,
    "PERSISTENT": Sector.IT,

    # Pharma
    "SUNPHARMA": Sector.PHARMA,
    "DRREDDY": Sector.PHARMA,
    "CIPLA": Sector.PHARMA,
    "DIVISLAB": Sector.PHARMA,
    "BIOCON": Sector.PHARMA,
    "LUPIN": Sector.PHARMA,
    "AUROPHARMA": Sector.PHARMA,
    "TORNTPHARM": Sector.PHARMA,

    # Auto
    "TATAMOTORS": Sector.AUTO,
    "MARUTI": Sector.AUTO,
    "M&M": Sector.AUTO,
    "BAJAJ-AUTO": Sector.AUTO,
    "HEROMOTOCO": Sector.AUTO,
    "EICHERMOT": Sector.AUTO,
    "ASHOKLEY": Sector.AUTO,
    "TVSMOTOR": Sector.AUTO,

    # FMCG
    "HINDUNILVR": Sector.FMCG,
    "ITC": Sector.FMCG,
    "NESTLEIND": Sector.FMCG,
    "BRITANNIA": Sector.FMCG,
    "DABUR": Sector.FMCG,
    "MARICO": Sector.FMCG,
    "COLPAL": Sector.FMCG,
    "GODREJCP": Sector.FMCG,

    # Metals
    "TATASTEEL": Sector.METALS,
    "JSWSTEEL": Sector.METALS,
    "HINDALCO": Sector.METALS,
    "VEDL": Sector.METALS,
    "COALINDIA": Sector.METALS,
    "NMDC": Sector.METALS,
    "SAIL": Sector.METALS,
    "JINDALSTEL": Sector.METALS,

    # Energy
    "RELIANCE": Sector.ENERGY,
    "ONGC": Sector.ENERGY,
    "BPCL": Sector.ENERGY,
    "IOC": Sector.ENERGY,
    "GAIL": Sector.ENERGY,
    "NTPC": Sector.ENERGY,
    "POWERGRID": Sector.ENERGY,
    "ADANIGREEN": Sector.ENERGY,

    # Infrastructure
    "LT": Sector.INFRASTRUCTURE,
    "ADANIENT": Sector.INFRASTRUCTURE,
    "ADANIPORTS": Sector.INFRASTRUCTURE,
    "GRASIM": Sector.INFRASTRUCTURE,
    "ULTRACEMCO": Sector.CEMENT,
    "SHREECEM": Sector.CEMENT,
    "AMBUJACEM": Sector.CEMENT,

    # Telecom
    "BHARTIARTL": Sector.TELECOM,
    "IDEA": Sector.TELECOM,

    # Financial Services
    "BAJFINANCE": Sector.FINANCIAL_SERVICES,
    "BAJAJFINSV": Sector.FINANCIAL_SERVICES,
    "HDFC": Sector.FINANCIAL_SERVICES,
    "SBILIFE": Sector.INSURANCE,
    "HDFCLIFE": Sector.INSURANCE,
    "ICICIPRULI": Sector.INSURANCE,

    # Consumer Durables
    "TITAN": Sector.CONSUMER_DURABLES,
    "HAVELLS": Sector.CONSUMER_DURABLES,
    "VOLTAS": Sector.CONSUMER_DURABLES,
    "WHIRLPOOL": Sector.CONSUMER_DURABLES,

    # Healthcare
    "APOLLOHOSP": Sector.HEALTHCARE,
    "FORTIS": Sector.HEALTHCARE,
    "MAXHEALTH": Sector.HEALTHCARE,

    # Realty
    "DLF": Sector.REALTY,
    "GODREJPROP": Sector.REALTY,
    "OBEROIRLTY": Sector.REALTY,
    "PRESTIGE": Sector.REALTY,
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PortfolioRiskConfig:
    """Configuration for portfolio risk management."""
    # Correlation limits
    max_correlation: float = 0.7            # Max allowed correlation with existing positions
    correlation_lookback: int = 60          # Days for correlation calculation

    # Sector exposure limits
    max_sector_exposure_pct: float = 30.0   # Max % of portfolio in one sector
    max_correlated_sector_pct: float = 50.0 # Max % in highly correlated sectors

    # Position limits
    max_single_position_pct: float = 15.0   # Max % in single position
    max_positions: int = 20                 # Max number of positions
    min_positions_for_diversification: int = 5  # Min positions to be "diversified"

    # Concentration limits
    max_top3_concentration: float = 50.0    # Max % in top 3 positions
    max_top5_concentration: float = 70.0    # Max % in top 5 positions

    # Beta limits
    max_portfolio_beta: float = 1.5         # Max portfolio beta
    min_portfolio_beta: float = 0.5         # Min portfolio beta (avoid too defensive)

    # Custom sector map
    sector_map: Dict[str, Sector] = field(default_factory=dict)


@dataclass
class PositionRisk:
    """Risk metrics for a single position."""
    symbol: str
    quantity: int
    current_price: float
    market_value: float
    weight_pct: float
    sector: Sector
    correlations: List[CorrelationPair] = field(default_factory=list)
    beta: float = 1.0


@dataclass
class SectorExposure:
    """Exposure metrics for a sector."""
    sector: Sector
    symbols: List[str]
    total_value: float
    weight_pct: float
    position_count: int

    @property
    def is_concentrated(self) -> bool:
        """Check if sector is over-concentrated (>30%)."""
        return self.weight_pct > 30.0


@dataclass
class PortfolioRiskReport:
    """Complete portfolio risk report."""
    timestamp: datetime
    total_value: float
    position_count: int

    # Sector analysis
    sector_exposures: List[SectorExposure]
    max_sector_exposure_pct: float
    sector_warnings: List[str]

    # Correlation analysis
    avg_correlation: float
    high_correlation_pairs: List[CorrelationPair]
    correlation_warnings: List[str]

    # Concentration
    top3_concentration: float
    top5_concentration: float
    herfindahl_index: float  # Concentration measure

    # Diversification
    diversification_score: float  # 0-1, higher = better
    diversification_warnings: List[str]

    # Overall
    risk_score: float  # 0-100, higher = more risky
    is_healthy: bool
    all_warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Generate text summary."""
        lines = [
            "=" * 60,
            "PORTFOLIO RISK REPORT",
            "=" * 60,
            f"Timestamp: {self.timestamp.strftime('%Y-%m-%d %H:%M')}",
            f"Total Value: Rs.{self.total_value:,.0f}",
            f"Positions: {self.position_count}",
            "",
            "DIVERSIFICATION",
            f"  Score: {self.diversification_score:.1%}",
            f"  Avg Correlation: {self.avg_correlation:.2f}",
            f"  Top 3 Concentration: {self.top3_concentration:.1f}%",
            "",
            "SECTOR EXPOSURE",
            f"  Max Sector: {self.max_sector_exposure_pct:.1f}%",
        ]

        for exp in sorted(self.sector_exposures, key=lambda x: -x.weight_pct)[:5]:
            lines.append(f"    {exp.sector.value}: {exp.weight_pct:.1f}%")

        lines.extend([
            "",
            f"RISK SCORE: {self.risk_score:.0f}/100",
            f"Status: {'HEALTHY' if self.is_healthy else 'WARNING'}",
        ])

        if self.all_warnings:
            lines.append("\nWARNINGS:")
            for w in self.all_warnings:
                lines.append(f"  - {w}")

        lines.append("=" * 60)
        return "\n".join(lines)


# =============================================================================
# PORTFOLIO RISK MANAGER
# =============================================================================

class PortfolioRiskManager:
    """
    Portfolio-level risk management.

    Monitors and controls:
    - Correlation between positions
    - Sector exposure
    - Position concentration
    - Portfolio diversification
    """

    def __init__(
        self,
        config: Optional[PortfolioRiskConfig] = None,
        price_data: Optional[pd.DataFrame] = None
    ):
        self.config = config or PortfolioRiskConfig()
        self.price_data = price_data
        self.correlation_analyzer = CorrelationAnalyzer()

        # Merge custom sector map with defaults
        self.sector_map = {**DEFAULT_SECTOR_MAP, **self.config.sector_map}

        # Cache
        self._correlation_matrix: Optional[CorrelationMatrix] = None
        self._last_correlation_update: Optional[datetime] = None

    def set_price_data(self, price_data: pd.DataFrame) -> None:
        """Update price data for correlation analysis."""
        self.price_data = price_data
        self._correlation_matrix = None  # Invalidate cache

    def get_sector(self, symbol: str) -> Sector:
        """Get sector for a symbol."""
        return self.sector_map.get(symbol.upper(), Sector.UNKNOWN)

    def add_sector_mapping(self, symbol: str, sector: Sector) -> None:
        """Add or update sector mapping for a symbol."""
        self.sector_map[symbol.upper()] = sector

    def _get_correlation_matrix(self) -> Optional[CorrelationMatrix]:
        """Get or calculate correlation matrix."""
        if self.price_data is None:
            return None

        # Check cache
        if self._correlation_matrix is not None:
            return self._correlation_matrix

        self._correlation_matrix = self.correlation_analyzer.calculate_correlation_matrix(
            self.price_data,
            lookback=self.config.correlation_lookback
        )
        self._last_correlation_update = datetime.now()

        return self._correlation_matrix

    def check_correlation_risk(
        self,
        new_symbol: str,
        existing_positions: Dict[str, float]  # symbol -> value
    ) -> Tuple[bool, List[str]]:
        """
        Check if adding a position would create correlation risk.

        Args:
            new_symbol: Symbol to potentially add
            existing_positions: Current positions {symbol: market_value}

        Returns:
            (is_safe, list of warnings)
        """
        warnings = []

        if self.price_data is None:
            return True, ["No price data for correlation check"]

        existing_symbols = list(existing_positions.keys())
        if not existing_symbols:
            return True, []

        is_safe, high_corr = self.correlation_analyzer.check_new_position_correlation(
            new_symbol,
            existing_symbols,
            self.price_data,
            threshold=self.config.max_correlation
        )

        for pair in high_corr:
            warnings.append(
                f"High correlation ({pair.correlation:.2f}) with {pair.symbol2}"
            )

        return is_safe, warnings

    def check_sector_exposure(
        self,
        new_symbol: str,
        new_value: float,
        existing_positions: Dict[str, float]
    ) -> Tuple[bool, List[str]]:
        """
        Check if adding position would exceed sector limits.

        Args:
            new_symbol: Symbol to add
            new_value: Value of new position
            existing_positions: Current positions

        Returns:
            (is_safe, list of warnings)
        """
        warnings = []

        new_sector = self.get_sector(new_symbol)

        # Calculate current sector exposure
        sector_values: Dict[Sector, float] = {}
        for symbol, value in existing_positions.items():
            sector = self.get_sector(symbol)
            sector_values[sector] = sector_values.get(sector, 0) + value

        # Add new position
        sector_values[new_sector] = sector_values.get(new_sector, 0) + new_value

        # Calculate total
        total_value = sum(sector_values.values())
        if total_value <= 0:
            return True, []

        # Check sector exposure
        new_sector_pct = (sector_values[new_sector] / total_value) * 100

        if new_sector_pct > self.config.max_sector_exposure_pct:
            warnings.append(
                f"Sector {new_sector.value} would be {new_sector_pct:.1f}% "
                f"(limit: {self.config.max_sector_exposure_pct}%)"
            )
            return False, warnings

        return True, warnings

    def check_concentration_risk(
        self,
        new_symbol: str,
        new_value: float,
        existing_positions: Dict[str, float]
    ) -> Tuple[bool, List[str]]:
        """
        Check if adding position would create concentration risk.

        Args:
            new_symbol: Symbol to add
            new_value: Value of new position
            existing_positions: Current positions

        Returns:
            (is_safe, list of warnings)
        """
        warnings = []

        # Combine positions
        all_positions = {**existing_positions, new_symbol: new_value}
        total_value = sum(all_positions.values())

        if total_value <= 0:
            return True, []

        # Check single position limit
        new_pct = (new_value / total_value) * 100
        if new_pct > self.config.max_single_position_pct:
            warnings.append(
                f"Position would be {new_pct:.1f}% "
                f"(limit: {self.config.max_single_position_pct}%)"
            )
            return False, warnings

        # Check top 3 concentration
        sorted_values = sorted(all_positions.values(), reverse=True)
        top3_pct = (sum(sorted_values[:3]) / total_value) * 100

        if top3_pct > self.config.max_top3_concentration:
            warnings.append(
                f"Top 3 concentration would be {top3_pct:.1f}% "
                f"(limit: {self.config.max_top3_concentration}%)"
            )

        return len(warnings) == 0, warnings

    def can_add_position(
        self,
        symbol: str,
        value: float,
        existing_positions: Dict[str, float]
    ) -> Tuple[bool, List[str]]:
        """
        Complete check if a position can be added.

        Args:
            symbol: Symbol to add
            value: Value of position
            existing_positions: Current portfolio

        Returns:
            (can_add, list of all warnings)
        """
        all_warnings = []

        # Check max positions
        if len(existing_positions) >= self.config.max_positions:
            all_warnings.append(f"Max positions reached ({self.config.max_positions})")
            return False, all_warnings

        # Check correlation
        corr_ok, corr_warnings = self.check_correlation_risk(symbol, existing_positions)
        all_warnings.extend(corr_warnings)

        # Check sector
        sector_ok, sector_warnings = self.check_sector_exposure(
            symbol, value, existing_positions
        )
        all_warnings.extend(sector_warnings)

        # Check concentration
        conc_ok, conc_warnings = self.check_concentration_risk(
            symbol, value, existing_positions
        )
        all_warnings.extend(conc_warnings)

        can_add = corr_ok and sector_ok and conc_ok
        return can_add, all_warnings

    def get_sector_exposures(
        self,
        positions: Dict[str, float]
    ) -> List[SectorExposure]:
        """Calculate exposure for each sector."""
        sector_data: Dict[Sector, Dict[str, Any]] = {}

        total_value = sum(positions.values())
        if total_value <= 0:
            return []

        for symbol, value in positions.items():
            sector = self.get_sector(symbol)

            if sector not in sector_data:
                sector_data[sector] = {
                    'symbols': [],
                    'total_value': 0.0
                }

            sector_data[sector]['symbols'].append(symbol)
            sector_data[sector]['total_value'] += value

        exposures = []
        for sector, data in sector_data.items():
            exposures.append(SectorExposure(
                sector=sector,
                symbols=data['symbols'],
                total_value=data['total_value'],
                weight_pct=(data['total_value'] / total_value) * 100,
                position_count=len(data['symbols'])
            ))

        return sorted(exposures, key=lambda x: -x.weight_pct)

    def calculate_diversification_score(
        self,
        positions: Dict[str, float]
    ) -> float:
        """
        Calculate portfolio diversification score.

        Score of 1.0 = perfectly diversified
        Score of 0.0 = completely concentrated

        Factors:
        - Number of positions
        - Sector spread
        - Correlation between positions
        - Concentration (Herfindahl index)
        """
        if len(positions) == 0:
            return 0.0

        scores = []

        # Position count score (more positions = better, up to a point)
        n = len(positions)
        min_n = self.config.min_positions_for_diversification
        position_score = min(1.0, n / min_n)
        scores.append(position_score)

        # Sector spread score
        sector_exposures = self.get_sector_exposures(positions)
        n_sectors = len([e for e in sector_exposures if e.weight_pct >= 5])
        sector_score = min(1.0, n_sectors / 5)  # 5 sectors = perfect
        scores.append(sector_score)

        # Concentration score (inverse of Herfindahl index)
        total_value = sum(positions.values())
        if total_value > 0:
            weights = [v / total_value for v in positions.values()]
            herfindahl = sum(w ** 2 for w in weights)
            # HHI of 1/n is perfectly equal weighted
            # HHI of 1 is completely concentrated
            hhi_score = 1 - herfindahl
            scores.append(hhi_score)

        # Correlation score
        if self.price_data is not None and len(positions) >= 2:
            symbols = list(positions.keys())
            available = [s for s in symbols if s in self.price_data.columns]
            if len(available) >= 2:
                corr_score = self.correlation_analyzer.get_diversification_score(
                    available, self.price_data
                )
                scores.append(corr_score)

        return np.mean(scores)

    def generate_risk_report(
        self,
        positions: Dict[str, float]
    ) -> PortfolioRiskReport:
        """
        Generate comprehensive portfolio risk report.

        Args:
            positions: Current positions {symbol: market_value}

        Returns:
            PortfolioRiskReport with all risk metrics
        """
        total_value = sum(positions.values())
        position_count = len(positions)
        all_warnings = []

        # Sector analysis
        sector_exposures = self.get_sector_exposures(positions)
        max_sector_pct = sector_exposures[0].weight_pct if sector_exposures else 0

        sector_warnings = []
        for exp in sector_exposures:
            if exp.weight_pct > self.config.max_sector_exposure_pct:
                warning = f"Sector {exp.sector.value} over-exposed at {exp.weight_pct:.1f}%"
                sector_warnings.append(warning)
                all_warnings.append(warning)

        # Correlation analysis
        high_correlations = []
        avg_correlation = 0.0
        correlation_warnings = []

        if self.price_data is not None and len(positions) >= 2:
            matrix = self._get_correlation_matrix()
            if matrix:
                symbols = [s for s in positions.keys() if s in matrix.symbols]
                if len(symbols) >= 2:
                    sub_matrix = CorrelationMatrix(
                        matrix.matrix.loc[symbols, symbols],
                        matrix.lookback_days,
                        matrix.timestamp
                    )
                    high_correlations = sub_matrix.get_high_correlations(
                        self.config.max_correlation
                    )
                    avg_correlation = sub_matrix.average_correlation()

                    for pair in high_correlations[:3]:
                        warning = f"High correlation: {pair.symbol1}-{pair.symbol2} ({pair.correlation:.2f})"
                        correlation_warnings.append(warning)
                        all_warnings.append(warning)

        # Concentration
        sorted_values = sorted(positions.values(), reverse=True)
        top3_conc = (sum(sorted_values[:3]) / total_value * 100) if total_value > 0 else 0
        top5_conc = (sum(sorted_values[:5]) / total_value * 100) if total_value > 0 else 0

        # Herfindahl index
        if total_value > 0:
            weights = [v / total_value for v in positions.values()]
            herfindahl = sum(w ** 2 for w in weights)
        else:
            herfindahl = 0

        # Diversification
        diversification_score = self.calculate_diversification_score(positions)
        div_warnings = []
        if diversification_score < 0.5:
            warning = f"Low diversification score: {diversification_score:.1%}"
            div_warnings.append(warning)
            all_warnings.append(warning)

        if position_count < self.config.min_positions_for_diversification:
            warning = f"Only {position_count} positions (recommend {self.config.min_positions_for_diversification}+)"
            div_warnings.append(warning)
            all_warnings.append(warning)

        # Calculate overall risk score (0-100)
        risk_factors = []

        # Sector concentration (0-40)
        risk_factors.append(min(40, max_sector_pct * 0.8))

        # Correlation (0-30)
        if avg_correlation > 0:
            risk_factors.append(min(30, avg_correlation * 30))

        # Position concentration (0-30)
        risk_factors.append(min(30, herfindahl * 30))

        risk_score = sum(risk_factors)
        is_healthy = risk_score < 50 and len(all_warnings) <= 2

        return PortfolioRiskReport(
            timestamp=datetime.now(),
            total_value=total_value,
            position_count=position_count,
            sector_exposures=sector_exposures,
            max_sector_exposure_pct=max_sector_pct,
            sector_warnings=sector_warnings,
            avg_correlation=avg_correlation,
            high_correlation_pairs=high_correlations,
            correlation_warnings=correlation_warnings,
            top3_concentration=top3_conc,
            top5_concentration=top5_conc,
            herfindahl_index=herfindahl,
            diversification_score=diversification_score,
            diversification_warnings=div_warnings,
            risk_score=risk_score,
            is_healthy=is_healthy,
            all_warnings=all_warnings
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def check_correlation_risk(
    new_symbol: str,
    existing_symbols: List[str],
    price_data: pd.DataFrame,
    threshold: float = 0.7
) -> Tuple[bool, List[CorrelationPair]]:
    """Quick correlation risk check."""
    analyzer = CorrelationAnalyzer()
    return analyzer.check_new_position_correlation(
        new_symbol, existing_symbols, price_data, threshold
    )


def check_sector_exposure(
    positions: Dict[str, float],
    max_pct: float = 30.0
) -> Tuple[bool, List[SectorExposure]]:
    """Quick sector exposure check."""
    mgr = PortfolioRiskManager()
    exposures = mgr.get_sector_exposures(positions)
    violations = [e for e in exposures if e.weight_pct > max_pct]
    return len(violations) == 0, violations
