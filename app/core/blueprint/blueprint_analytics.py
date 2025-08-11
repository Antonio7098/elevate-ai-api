"""
Blueprint analytics module.

This module provides analytics functionality for blueprints.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from app.models.blueprint import Blueprint, BlueprintStatus, BlueprintType
from app.core.blueprint.blueprint_repository import BlueprintRepository
from app.core.blueprint.blueprint_indexer import BlueprintIndexer
import json


class BlueprintAnalytics:
    """Analytics class for blueprint metrics and insights."""
    
    def __init__(self, repository: BlueprintRepository, indexer: BlueprintIndexer):
        self.repository = repository
        self.indexer = indexer
        self.usage_tracking: Dict[str, Dict[str, Any]] = {}
    
    async def track_blueprint_usage(self, blueprint_id: str, action: str, user_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Track usage of a blueprint."""
        if blueprint_id not in self.usage_tracking:
            self.usage_tracking[blueprint_id] = {
                'views': 0,
                'searches': 0,
                'downloads': 0,
                'shares': 0,
                'last_accessed': None,
                'user_actions': []
            }
        
        # Update usage counts
        if action in ['view', 'open', 'read']:
            self.usage_tracking[blueprint_id]['views'] += 1
        elif action in ['search', 'query']:
            self.usage_tracking[blueprint_id]['searches'] += 1
        elif action in ['download', 'export']:
            self.usage_tracking[blueprint_id]['downloads'] += 1
        elif action in ['share', 'link']:
            self.usage_tracking[blueprint_id]['shares'] += 1
        
        # Update last accessed time
        self.usage_tracking[blueprint_id]['last_accessed'] = datetime.utcnow().isoformat()
        
        # Track user action
        user_action = {
            'timestamp': datetime.utcnow().isoformat(),
            'action': action,
            'user_id': user_id,
            'metadata': metadata or {}
        }
        self.usage_tracking[blueprint_id]['user_actions'].append(user_action)
        
        # Keep only last 100 actions to prevent memory bloat
        if len(self.usage_tracking[blueprint_id]['user_actions']) > 100:
            self.usage_tracking[blueprint_id]['user_actions'] = self.usage_tracking[blueprint_id]['user_actions'][-100:]
    
    async def get_blueprint_analytics(self, blueprint_id: str) -> Dict[str, Any]:
        """Get analytics for a specific blueprint."""
        # Get basic usage stats
        usage_stats = self.usage_tracking.get(blueprint_id, {})
        
        # Get blueprint info
        blueprint = await self.repository.get_by_id(blueprint_id)
        if not blueprint:
            return {'error': 'Blueprint not found'}
        
        # Calculate engagement metrics
        total_actions = sum([
            usage_stats.get('views', 0),
            usage_stats.get('searches', 0),
            usage_stats.get('downloads', 0),
            usage_stats.get('shares', 0)
        ])
        
        # Calculate recent activity (last 30 days)
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        recent_actions = [
            action for action in usage_stats.get('user_actions', [])
            if datetime.fromisoformat(action['timestamp']) > thirty_days_ago
        ]
        
        return {
            'blueprint_id': blueprint_id,
            'title': blueprint.title,
            'type': blueprint.type.value,
            'status': blueprint.status.value,
            'created_at': blueprint.created_at.isoformat(),
            'updated_at': blueprint.updated_at.isoformat(),
            'usage_stats': {
                'total_views': usage_stats.get('views', 0),
                'total_searches': usage_stats.get('searches', 0),
                'total_downloads': usage_stats.get('downloads', 0),
                'total_shares': usage_stats.get('shares', 0),
                'total_actions': total_actions,
                'recent_actions_30d': len(recent_actions),
                'last_accessed': usage_stats.get('last_accessed')
            },
            'engagement_metrics': {
                'engagement_score': self._calculate_engagement_score(usage_stats),
                'popularity_rank': await self._calculate_popularity_rank(blueprint_id),
                'activity_trend': self._calculate_activity_trend(usage_stats.get('user_actions', []))
            }
        }
    
    async def get_system_analytics(self) -> Dict[str, Any]:
        """Get system-wide analytics for all blueprints."""
        # Get all blueprints
        all_blueprints = await self.repository.list_all(limit=10000, offset=0)
        
        # Calculate system metrics
        total_blueprints = len(all_blueprints)
        blueprints_by_status = {}
        blueprints_by_type = {}
        total_views = 0
        total_searches = 0
        total_downloads = 0
        total_shares = 0
        
        for blueprint in all_blueprints:
            # Count by status
            status = blueprint.status.value
            blueprints_by_status[status] = blueprints_by_status.get(status, 0) + 1
            
            # Count by type
            blueprint_type = blueprint.type.value
            blueprints_by_type[blueprint_type] = blueprints_by_type.get(blueprint_type, 0) + 1
            
            # Sum usage stats
            usage = self.usage_tracking.get(blueprint.id, {})
            total_views += usage.get('views', 0)
            total_searches += usage.get('searches', 0)
            total_downloads += usage.get('downloads', 0)
            total_shares += usage.get('shares', 0)
        
        # Calculate averages
        avg_views_per_blueprint = total_views / total_blueprints if total_blueprints > 0 else 0
        avg_searches_per_blueprint = total_searches / total_blueprints if total_blueprints > 0 else 0
        
        return {
            'system_overview': {
                'total_blueprints': total_blueprints,
                'total_views': total_views,
                'total_searches': total_searches,
                'total_downloads': total_downloads,
                'total_shares': total_shares
            },
            'blueprint_distribution': {
                'by_status': blueprints_by_status,
                'by_type': blueprints_by_type
            },
            'performance_metrics': {
                'avg_views_per_blueprint': round(avg_views_per_blueprint, 2),
                'avg_searches_per_blueprint': round(avg_searches_per_blueprint, 2),
                'most_popular_blueprints': await self._get_most_popular_blueprints(5),
                'most_active_blueprints': await self._get_most_active_blueprints(5)
            },
            'trends': {
                'recent_activity': await self._get_recent_activity_summary(),
                'growth_rate': await self._calculate_growth_rate()
            }
        }
    
    async def get_user_analytics(self, user_id: str) -> Dict[str, Any]:
        """Get analytics for a specific user."""
        # Get user's blueprints
        user_blueprints = await self.repository.get_by_author(user_id, limit=1000, offset=0)
        
        # Calculate user metrics
        total_blueprints = len(user_blueprints)
        total_views = 0
        total_searches = 0
        total_downloads = 0
        total_shares = 0
        
        for blueprint in user_blueprints:
            usage = self.usage_tracking.get(blueprint.id, {})
            total_views += usage.get('views', 0)
            total_searches += usage.get('searches', 0)
            total_downloads += usage.get('downloads', 0)
            total_shares += usage.get('shares', 0)
        
        # Get user's most popular blueprint
        most_popular = None
        max_views = 0
        for blueprint in user_blueprints:
            usage = self.usage_tracking.get(blueprint.id, {})
            if usage.get('views', 0) > max_views:
                max_views = usage.get('views', 0)
                most_popular = blueprint
        
        return {
            'user_id': user_id,
            'blueprint_count': total_blueprints,
            'usage_summary': {
                'total_views': total_views,
                'total_searches': total_searches,
                'total_downloads': total_downloads,
                'total_shares': total_shares
            },
            'performance': {
                'most_popular_blueprint': {
                    'id': most_popular.id if most_popular else None,
                    'title': most_popular.title if most_popular else None,
                    'views': max_views
                },
                'avg_views_per_blueprint': round(total_views / total_blueprints, 2) if total_blueprints > 0 else 0
            },
            'recent_activity': await self._get_user_recent_activity(user_id)
        }
    
    def _calculate_engagement_score(self, usage_stats: Dict[str, Any]) -> float:
        """Calculate engagement score based on usage patterns."""
        views = usage_stats.get('views', 0)
        searches = usage_stats.get('searches', 0)
        downloads = usage_stats.get('downloads', 0)
        shares = usage_stats.get('shares', 0)
        
        # Weighted scoring: views (1x), searches (2x), downloads (3x), shares (4x)
        score = views + (searches * 2) + (downloads * 3) + (shares * 4)
        
        # Normalize to 0-100 scale
        normalized_score = min(score / 10, 100)
        return round(normalized_score, 2)
    
    async def _calculate_popularity_rank(self, blueprint_id: str) -> int:
        """Calculate popularity rank of a blueprint among all blueprints."""
        all_blueprints = await self.repository.list_all(limit=10000, offset=0)
        
        # Calculate engagement scores for all blueprints
        blueprint_scores = []
        for blueprint in all_blueprints:
            usage = self.usage_tracking.get(blueprint.id, {})
            score = self._calculate_engagement_score(usage)
            blueprint_scores.append((blueprint.id, score))
        
        # Sort by score and find rank
        blueprint_scores.sort(key=lambda x: x[1], reverse=True)
        
        for rank, (bid, _) in enumerate(blueprint_scores, 1):
            if bid == blueprint_id:
                return rank
        
        return len(blueprint_scores) + 1
    
    def _calculate_activity_trend(self, user_actions: List[Dict[str, Any]]) -> str:
        """Calculate activity trend based on recent user actions."""
        if not user_actions:
            return 'no_activity'
        
        # Group actions by day for last 7 days
        seven_days_ago = datetime.utcnow() - timedelta(days=7)
        daily_actions = {}
        
        for action in user_actions:
            action_date = datetime.fromisoformat(action['timestamp']).date()
            if action_date >= seven_days_ago.date():
                daily_actions[action_date] = daily_actions.get(action_date, 0) + 1
        
        if not daily_actions:
            return 'no_recent_activity'
        
        # Calculate trend
        dates = sorted(daily_actions.keys())
        if len(dates) < 2:
            return 'insufficient_data'
        
        recent_avg = sum(daily_actions[date] for date in dates[-3:]) / 3
        earlier_avg = sum(daily_actions[date] for date in dates[:len(dates)-3]) / max(len(dates)-3, 1)
        
        if recent_avg > earlier_avg * 1.2:
            return 'increasing'
        elif recent_avg < earlier_avg * 0.8:
            return 'decreasing'
        else:
            return 'stable'
    
    async def _get_most_popular_blueprints(self, limit: int) -> List[Dict[str, Any]]:
        """Get most popular blueprints by engagement score."""
        all_blueprints = await self.repository.list_all(limit=10000, offset=0)
        
        blueprint_scores = []
        for blueprint in all_blueprints:
            usage = self.usage_tracking.get(blueprint.id, {})
            score = self._calculate_engagement_score(usage)
            blueprint_scores.append((blueprint, score))
        
        # Sort by score and return top results
        blueprint_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [
            {
                'id': blueprint.id,
                'title': blueprint.title,
                'type': blueprint.type.value,
                'engagement_score': score,
                'views': self.usage_tracking.get(blueprint.id, {}).get('views', 0)
            }
            for blueprint, score in blueprint_scores[:limit]
        ]
    
    async def _get_most_active_blueprints(self, limit: int) -> List[Dict[str, Any]]:
        """Get most active blueprints by recent activity."""
        all_blueprints = await self.repository.list_all(limit=10000, offset=0)
        
        blueprint_activity = []
        for blueprint in all_blueprints:
            usage = self.usage_tracking.get(blueprint.id, {})
            recent_actions = [
                action for action in usage.get('user_actions', [])
                if datetime.fromisoformat(action['timestamp']) > datetime.utcnow() - timedelta(days=7)
            ]
            blueprint_activity.append((blueprint, len(recent_actions)))
        
        # Sort by recent activity and return top results
        blueprint_activity.sort(key=lambda x: x[1], reverse=True)
        
        return [
            {
                'id': blueprint.id,
                'title': blueprint.title,
                'type': blueprint.type.value,
                'recent_activity_7d': activity_count
            }
            for blueprint, activity_count in blueprint_activity[:limit]
        ]
    
    async def _get_recent_activity_summary(self) -> Dict[str, Any]:
        """Get summary of recent activity across all blueprints."""
        # This is a simplified implementation
        # In a real system, you might track more detailed activity patterns
        
        return {
            'last_24h': 'data_not_available',
            'last_7d': 'data_not_available',
            'last_30d': 'data_not_available'
        }
    
    async def _calculate_growth_rate(self) -> str:
        """Calculate growth rate of blueprint creation."""
        # This is a simplified implementation
        # In a real system, you would compare creation rates over time periods
        
        return 'data_not_available'
    
    async def _get_user_recent_activity(self, user_id: str) -> List[Dict[str, Any]]:
        """Get recent activity for a specific user."""
        # This is a simplified implementation
        # In a real system, you would track user-specific activity patterns
        
        return [
            {
                'action': 'data_not_available',
                'timestamp': 'data_not_available',
                'blueprint_id': 'data_not_available'
            }
        ]
    
    async def export_analytics_data(self, format: str = 'json') -> str:
        """Export analytics data in specified format."""
        if format.lower() == 'json':
            data = {
                'usage_tracking': self.usage_tracking,
                'exported_at': datetime.utcnow().isoformat()
            }
            return json.dumps(data, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    async def clear_analytics_data(self) -> None:
        """Clear all analytics data."""
        self.usage_tracking.clear()
