"""
Advanced performance monitoring system for premium operations.
Implements comprehensive monitoring, cost efficiency tracking, and alerting.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
import json

@dataclass
class PerformanceMetrics:
    """Performance metrics for premium operations"""
    operation: str
    user_id: str
    latency_ms: float
    cost: float
    quality_score: float
    model_used: str
    cache_hit: bool
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class PerformanceReport:
    """Comprehensive performance report"""
    time_range: str
    total_operations: int
    avg_latency_ms: float
    total_cost: float
    avg_quality_score: float
    cache_hit_rate: float
    cost_efficiency_score: float
    alerts: List[Dict[str, Any]]
    recommendations: List[str]
    timestamp: datetime

@dataclass
class Alert:
    """Alert for monitoring system"""
    alert_id: str
    alert_type: str
    severity: str
    message: str
    user_id: Optional[str]
    operation: Optional[str]
    threshold: float
    current_value: float
    timestamp: datetime
    resolved: bool = False

class MetricsCollector:
    """Collects and stores performance metrics"""
    
    def __init__(self):
        self.metrics_history = []
        self.user_metrics = {}
        self.operation_metrics = {}
    
    async def collect_metric(self, metric: PerformanceMetrics):
        """Collect a performance metric"""
        try:
            self.metrics_history.append(metric)
            
            # Track per-user metrics
            if metric.user_id not in self.user_metrics:
                self.user_metrics[metric.user_id] = []
            self.user_metrics[metric.user_id].append(metric)
            
            # Track per-operation metrics
            if metric.operation not in self.operation_metrics:
                self.operation_metrics[metric.operation] = []
            self.operation_metrics[metric.operation].append(metric)
            
        except Exception as e:
            print(f"Error collecting metric: {e}")
    
    def get_user_metrics(self, user_id: str, time_range: str = "24h") -> List[PerformanceMetrics]:
        """Get metrics for a specific user"""
        try:
            if user_id not in self.user_metrics:
                return []
            
            cutoff_time = self._get_cutoff_time(time_range)
            return [
                metric for metric in self.user_metrics[user_id]
                if metric.timestamp >= cutoff_time
            ]
            
        except Exception as e:
            print(f"Error getting user metrics: {e}")
            return []
    
    def get_operation_metrics(self, operation: str, time_range: str = "24h") -> List[PerformanceMetrics]:
        """Get metrics for a specific operation"""
        try:
            if operation not in self.operation_metrics:
                return []
            
            cutoff_time = self._get_cutoff_time(time_range)
            return [
                metric for metric in self.operation_metrics[operation]
                if metric.timestamp >= cutoff_time
            ]
            
        except Exception as e:
            print(f"Error getting operation metrics: {e}")
            return []
    
    def _get_cutoff_time(self, time_range: str) -> datetime:
        """Get cutoff time based on time range"""
        now = datetime.utcnow()
        
        if time_range == "1h":
            return now - timedelta(hours=1)
        elif time_range == "6h":
            return now - timedelta(hours=6)
        elif time_range == "24h":
            return now - timedelta(hours=24)
        elif time_range == "7d":
            return now - timedelta(days=7)
        elif time_range == "30d":
            return now - timedelta(days=30)
        else:
            return now - timedelta(hours=24)  # Default to 24h

class PerformanceAnalyzer:
    """Analyzes performance metrics and identifies issues"""
    
    def __init__(self):
        self.thresholds = {
            'latency_ms': 5000,  # 5 seconds
            'cost_per_request': 0.10,  # $0.10 per request
            'quality_score': 0.7,  # 70% quality threshold
            'cache_hit_rate': 0.3,  # 30% cache hit rate
            'error_rate': 0.05  # 5% error rate
        }
    
    async def analyze_performance(self, metrics: List[PerformanceMetrics]) -> Dict[str, Any]:
        """Analyze performance metrics and identify issues"""
        try:
            if not metrics:
                return {
                    'status': 'no_data',
                    'issues': [],
                    'recommendations': []
                }
            
            analysis = {
                'total_operations': len(metrics),
                'avg_latency_ms': sum(m.latency_ms for m in metrics) / len(metrics),
                'total_cost': sum(m.cost for m in metrics),
                'avg_quality_score': sum(m.quality_score for m in metrics) / len(metrics),
                'cache_hit_rate': sum(1 for m in metrics if m.cache_hit) / len(metrics),
                'cost_per_request': sum(m.cost for m in metrics) / len(metrics),
                'issues': [],
                'recommendations': []
            }
            
            # Identify issues
            issues = await self._identify_issues(analysis, metrics)
            analysis['issues'] = issues
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(analysis, metrics)
            analysis['recommendations'] = recommendations
            
            return analysis
            
        except Exception as e:
            print(f"Error analyzing performance: {e}")
            return {
                'status': 'error',
                'issues': [],
                'recommendations': []
            }
    
    async def _identify_issues(self, analysis: Dict[str, Any], metrics: List[PerformanceMetrics]) -> List[Dict[str, Any]]:
        """Identify performance issues"""
        issues = []
        
        try:
            # Check latency issues
            if analysis['avg_latency_ms'] > self.thresholds['latency_ms']:
                issues.append({
                    'type': 'high_latency',
                    'severity': 'warning',
                    'message': f"Average latency ({analysis['avg_latency_ms']:.1f}ms) exceeds threshold ({self.thresholds['latency_ms']}ms)",
                    'current_value': analysis['avg_latency_ms'],
                    'threshold': self.thresholds['latency_ms']
                })
            
            # Check cost issues
            if analysis['cost_per_request'] > self.thresholds['cost_per_request']:
                issues.append({
                    'type': 'high_cost',
                    'severity': 'warning',
                    'message': f"Average cost per request (${analysis['cost_per_request']:.3f}) exceeds threshold (${self.thresholds['cost_per_request']})",
                    'current_value': analysis['cost_per_request'],
                    'threshold': self.thresholds['cost_per_request']
                })
            
            # Check quality issues
            if analysis['avg_quality_score'] < self.thresholds['quality_score']:
                issues.append({
                    'type': 'low_quality',
                    'severity': 'critical',
                    'message': f"Average quality score ({analysis['avg_quality_score']:.2f}) below threshold ({self.thresholds['quality_score']})",
                    'current_value': analysis['avg_quality_score'],
                    'threshold': self.thresholds['quality_score']
                })
            
            # Check cache hit rate issues
            if analysis['cache_hit_rate'] < self.thresholds['cache_hit_rate']:
                issues.append({
                    'type': 'low_cache_hit_rate',
                    'severity': 'info',
                    'message': f"Cache hit rate ({analysis['cache_hit_rate']:.2f}) below threshold ({self.thresholds['cache_hit_rate']})",
                    'current_value': analysis['cache_hit_rate'],
                    'threshold': self.thresholds['cache_hit_rate']
                })
            
        except Exception as e:
            print(f"Error identifying issues: {e}")
        
        return issues
    
    async def _generate_recommendations(self, analysis: Dict[str, Any], metrics: List[PerformanceMetrics]) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        try:
            # High latency recommendations
            if analysis['avg_latency_ms'] > self.thresholds['latency_ms']:
                recommendations.append("Consider implementing response caching to reduce latency")
                recommendations.append("Review model selection strategy to use faster models for simple queries")
            
            # High cost recommendations
            if analysis['cost_per_request'] > self.thresholds['cost_per_request']:
                recommendations.append("Implement model cascading to use cheaper models when possible")
                recommendations.append("Enable intelligent caching to reduce redundant API calls")
            
            # Low quality recommendations
            if analysis['avg_quality_score'] < self.thresholds['quality_score']:
                recommendations.append("Consider using more powerful models for complex queries")
                recommendations.append("Review prompt engineering to improve response quality")
            
            # Low cache hit rate recommendations
            if analysis['cache_hit_rate'] < self.thresholds['cache_hit_rate']:
                recommendations.append("Expand cache size and optimize cache eviction policies")
                recommendations.append("Implement semantic caching for similar queries")
            
        except Exception as e:
            print(f"Error generating recommendations: {e}")
        
        return recommendations

class AlertManager:
    """Manages alerts and notifications"""
    
    def __init__(self):
        self.alerts = []
        self.alert_rules = {
            'high_latency': {'threshold': 5000, 'severity': 'warning'},
            'high_cost': {'threshold': 0.10, 'severity': 'warning'},
            'low_quality': {'threshold': 0.7, 'severity': 'critical'},
            'low_cache_hit_rate': {'threshold': 0.3, 'severity': 'info'},
            'high_error_rate': {'threshold': 0.05, 'severity': 'critical'}
        }
    
    async def check_alerts(self, metrics: List[PerformanceMetrics]) -> List[Alert]:
        """Check for alert conditions"""
        try:
            new_alerts = []
            
            if not metrics:
                return new_alerts
            
            # Calculate current metrics
            avg_latency = sum(m.latency_ms for m in metrics) / len(metrics)
            total_cost = sum(m.cost for m in metrics)
            avg_quality = sum(m.quality_score for m in metrics) / len(metrics)
            cache_hit_rate = sum(1 for m in metrics if m.cache_hit) / len(metrics)
            cost_per_request = total_cost / len(metrics)
            
            # Check latency alert
            if avg_latency > self.alert_rules['high_latency']['threshold']:
                alert = Alert(
                    alert_id=f"latency_{datetime.utcnow().timestamp()}",
                    alert_type="high_latency",
                    severity=self.alert_rules['high_latency']['severity'],
                    message=f"Average latency ({avg_latency:.1f}ms) exceeds threshold",
                    user_id=None,
                    operation=None,
                    threshold=self.alert_rules['high_latency']['threshold'],
                    current_value=avg_latency,
                    timestamp=datetime.utcnow()
                )
                new_alerts.append(alert)
            
            # Check cost alert
            if cost_per_request > self.alert_rules['high_cost']['threshold']:
                alert = Alert(
                    alert_id=f"cost_{datetime.utcnow().timestamp()}",
                    alert_type="high_cost",
                    severity=self.alert_rules['high_cost']['severity'],
                    message=f"Average cost per request (${cost_per_request:.3f}) exceeds threshold",
                    user_id=None,
                    operation=None,
                    threshold=self.alert_rules['high_cost']['threshold'],
                    current_value=cost_per_request,
                    timestamp=datetime.utcnow()
                )
                new_alerts.append(alert)
            
            # Check quality alert
            if avg_quality < self.alert_rules['low_quality']['threshold']:
                alert = Alert(
                    alert_id=f"quality_{datetime.utcnow().timestamp()}",
                    alert_type="low_quality",
                    severity=self.alert_rules['low_quality']['severity'],
                    message=f"Average quality score ({avg_quality:.2f}) below threshold",
                    user_id=None,
                    operation=None,
                    threshold=self.alert_rules['low_quality']['threshold'],
                    current_value=avg_quality,
                    timestamp=datetime.utcnow()
                )
                new_alerts.append(alert)
            
            # Check cache hit rate alert
            if cache_hit_rate < self.alert_rules['low_cache_hit_rate']['threshold']:
                alert = Alert(
                    alert_id=f"cache_{datetime.utcnow().timestamp()}",
                    alert_type="low_cache_hit_rate",
                    severity=self.alert_rules['low_cache_hit_rate']['severity'],
                    message=f"Cache hit rate ({cache_hit_rate:.2f}) below threshold",
                    user_id=None,
                    operation=None,
                    threshold=self.alert_rules['low_cache_hit_rate']['threshold'],
                    current_value=cache_hit_rate,
                    timestamp=datetime.utcnow()
                )
                new_alerts.append(alert)
            
            # Add new alerts to history
            self.alerts.extend(new_alerts)
            
            return new_alerts
            
        except Exception as e:
            print(f"Error checking alerts: {e}")
            return []
    
    def get_active_alerts(self) -> List[Alert]:
        """Get active (unresolved) alerts"""
        return [alert for alert in self.alerts if not alert.resolved]
    
    def resolve_alert(self, alert_id: str):
        """Mark an alert as resolved"""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                break

class MonitoringDashboard:
    """Dashboard for monitoring data visualization"""
    
    def __init__(self):
        self.dashboard_data = {}
    
    async def update_dashboard(self, metrics: List[PerformanceMetrics], alerts: List[Alert]):
        """Update dashboard with latest data"""
        try:
            if not metrics:
                return
            
            # Calculate key metrics
            total_operations = len(metrics)
            avg_latency = sum(m.latency_ms for m in metrics) / total_operations
            total_cost = sum(m.cost for m in metrics)
            avg_quality = sum(m.quality_score for m in metrics) / total_operations
            cache_hit_rate = sum(1 for m in metrics if m.cache_hit) / total_operations
            
            # Model usage breakdown
            model_usage = {}
            for metric in metrics:
                model = metric.model_used
                if model not in model_usage:
                    model_usage[model] = {'count': 0, 'total_cost': 0}
                model_usage[model]['count'] += 1
                model_usage[model]['total_cost'] += metric.cost
            
            # Operation breakdown
            operation_usage = {}
            for metric in metrics:
                operation = metric.operation
                if operation not in operation_usage:
                    operation_usage[operation] = {'count': 0, 'avg_latency': 0}
                operation_usage[operation]['count'] += 1
                operation_usage[operation]['avg_latency'] += metric.latency_ms
            
            # Calculate average latency per operation
            for operation in operation_usage:
                operation_usage[operation]['avg_latency'] /= operation_usage[operation]['count']
            
            # Update dashboard data
            self.dashboard_data = {
                'overview': {
                    'total_operations': total_operations,
                    'avg_latency_ms': avg_latency,
                    'total_cost': total_cost,
                    'avg_quality_score': avg_quality,
                    'cache_hit_rate': cache_hit_rate,
                    'active_alerts': len([a for a in alerts if not a.resolved])
                },
                'model_usage': model_usage,
                'operation_usage': operation_usage,
                'recent_alerts': alerts[-10:] if alerts else [],  # Last 10 alerts
                'last_updated': datetime.utcnow()
            }
            
        except Exception as e:
            print(f"Error updating dashboard: {e}")
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data"""
        return self.dashboard_data

class PremiumMonitoringSystem:
    """Main premium monitoring system"""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.performance_analyzer = PerformanceAnalyzer()
        self.alert_manager = AlertManager()
        self.dashboard = MonitoringDashboard()
    
    async def track_premium_metrics(self, operation: str, metrics: Dict):
        """Track comprehensive metrics for premium operations"""
        try:
            performance_metric = PerformanceMetrics(
                operation=operation,
                user_id=metrics.get('user_id', 'unknown'),
                latency_ms=metrics.get('latency_ms', 0.0),
                cost=metrics.get('cost', 0.0),
                quality_score=metrics.get('quality_score', 0.0),
                model_used=metrics.get('model_used', 'unknown'),
                cache_hit=metrics.get('cache_hit', False),
                timestamp=datetime.utcnow(),
                metadata=metrics.get('metadata', {})
            )
            
            await self.metrics_collector.collect_metric(performance_metric)
            
        except Exception as e:
            print(f"Error tracking premium metrics: {e}")
    
    async def monitor_cost_efficiency(self, operation: str, cost: float, quality: float):
        """Monitor cost vs quality trade-offs"""
        try:
            # Calculate cost efficiency score
            cost_efficiency = quality / cost if cost > 0 else 0.0
            
            # Track cost efficiency metrics
            await self.track_premium_metrics(operation, {
                'cost': cost,
                'quality_score': quality,
                'cost_efficiency': cost_efficiency,
                'metadata': {'cost_efficiency_analysis': True}
            })
            
        except Exception as e:
            print(f"Error monitoring cost efficiency: {e}")
    
    async def generate_performance_report(self, time_range: str = "24h") -> PerformanceReport:
        """Generate comprehensive performance reports"""
        try:
            # Get metrics for the time range
            all_metrics = self.metrics_collector.metrics_history
            cutoff_time = self.metrics_collector._get_cutoff_time(time_range)
            recent_metrics = [m for m in all_metrics if m.timestamp >= cutoff_time]
            
            # Analyze performance
            analysis = await self.performance_analyzer.analyze_performance(recent_metrics)
            
            # Check for alerts
            alerts = await self.alert_manager.check_alerts(recent_metrics)
            
            # Update dashboard
            await self.dashboard.update_dashboard(recent_metrics, alerts)
            
            # Calculate cost efficiency score
            total_cost = sum(m.cost for m in recent_metrics)
            total_quality = sum(m.quality_score for m in recent_metrics)
            cost_efficiency_score = total_quality / total_cost if total_cost > 0 else 0.0
            
            return PerformanceReport(
                time_range=time_range,
                total_operations=len(recent_metrics),
                avg_latency_ms=analysis.get('avg_latency_ms', 0.0),
                total_cost=analysis.get('total_cost', 0.0),
                avg_quality_score=analysis.get('avg_quality_score', 0.0),
                cache_hit_rate=analysis.get('cache_hit_rate', 0.0),
                cost_efficiency_score=cost_efficiency_score,
                alerts=[{
                    'type': alert.alert_type,
                    'severity': alert.severity,
                    'message': alert.message,
                    'timestamp': alert.timestamp
                } for alert in alerts],
                recommendations=analysis.get('recommendations', []),
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            print(f"Error generating performance report: {e}")
            return PerformanceReport(
                time_range=time_range,
                total_operations=0,
                avg_latency_ms=0.0,
                total_cost=0.0,
                avg_quality_score=0.0,
                cache_hit_rate=0.0,
                cost_efficiency_score=0.0,
                alerts=[],
                recommendations=[],
                timestamp=datetime.utcnow()
            )
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data"""
        return self.dashboard.get_dashboard_data()
    
    def get_active_alerts(self) -> List[Alert]:
        """Get active alerts"""
        return self.alert_manager.get_active_alerts()











