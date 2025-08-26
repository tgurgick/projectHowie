# Multi-Model Guide for Howie CLI

## Overview

Howie Enhanced now supports **multiple AI models** with automatic task-based routing, similar to how Claude can switch between different models. This allows you to optimize for cost, speed, and capability based on your specific needs.

## Supported Models

### OpenAI Models
- **GPT-4o** - Premium model for complex analysis and reasoning
- **GPT-4o-mini** - Fast, cost-effective model for simple queries
- **GPT-4-turbo** - Balanced performance for general tasks

### Anthropic Models  
- **Claude 3 Opus** - Most capable for complex reasoning and analysis
- **Claude 3.5 Sonnet** - Excellent for code generation and balanced tasks
- **Claude 3 Haiku** - Fast model for simple classification tasks

### Perplexity Models
- **Perplexity Sonar** - Specialized for research with web access
- **Perplexity Sonar Pro** - Advanced research with comprehensive search

## Quick Start

### 1. Set up API Keys

```bash
# Required for different models
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export PERPLEXITY_API_KEY="pplx-..."
```

### 2. Start Enhanced CLI

```bash
# Use multi-model version
python howie_enhanced.py

# Or with specific default model
python howie_enhanced.py chat --model claude-3-5-sonnet
```

### 3. Automatic Model Selection

The system automatically chooses the best model:

```python
# Research queries → Perplexity
> Who got injured in yesterday's NFL games?

# Code generation → Claude Sonnet
> Generate a Python script to analyze my roster

# Complex analysis → GPT-4o
> Analyze the playoff implications for all NFC teams

# Simple queries → GPT-4o-mini
> List all Cowboys players
```

## Manual Model Selection

### Use Specific Model for One Query

```python
# Research with Perplexity
> @perplexity-sonar Who are the top waiver wire pickups this week?

# Code generation with Claude
> @claude-3-5-sonnet Create a function to calculate fantasy points

# Fast query with mini model
> @gpt-4o-mini Show me all QBs on my roster

# Complex analysis with premium model
> @claude-3-opus Detailed analysis of playoff scenarios
```

### Model Commands

```python
# View available models and usage
> model:info

# Switch default model
> model:switch claude-3-5-sonnet

# Configure task-specific models
> model:config research perplexity-sonar-pro

# Save configuration
> model:save
```

## Task-Based Model Routing

### Default Mappings

| Task Type | Default Model | Reason |
|-----------|---------------|---------|
| `research` | Perplexity Sonar | Web access, current information |
| `analysis` | GPT-4o | Strong reasoning capabilities |
| `code_generation` | Claude 3.5 Sonnet | Excellent code quality |
| `optimization` | GPT-4o | Complex mathematical reasoning |
| `simple_query` | GPT-4o-mini | Cost-effective for basic tasks |
| `classification` | Claude 3 Haiku | Fast and accurate |

### Custom Configuration

```json
{
  "task_mappings": {
    "research": "perplexity-sonar-pro",
    "analysis": "claude-3-opus",
    "code_generation": "claude-3-5-sonnet",
    "optimization": "gpt-4o",
    "simple_query": "gpt-4o-mini"
  }
}
```

## Agent Model Configuration

Each agent type has optimized model preferences:

```python
# Research agents use Perplexity
> Spawn a research agent to investigate cold weather game impacts

# Analysis agents use GPT-4o
> Spawn an analysis agent to compare top RBs

# Code agents use Claude Sonnet
> Spawn a code agent to create analysis scripts

# Optimization agents use GPT-4o
> Spawn an optimization agent for my lineup
```

## Cost Optimization

### Model Cost Comparison (per 1K tokens)

| Model | Input Cost | Output Cost | Best For |
|-------|------------|-------------|----------|
| GPT-4o-mini | $0.00015 | $0.0006 | Simple queries, classification |
| Claude 3 Haiku | $0.00025 | $0.00125 | Fast tasks, extraction |
| Perplexity Sonar | $0.0006 | $0.0018 | Research, current info |
| Claude 3.5 Sonnet | $0.003 | $0.015 | Code generation, analysis |
| GPT-4o | $0.005 | $0.015 | Complex reasoning |
| Claude 3 Opus | $0.015 | $0.075 | Most complex tasks |

### Cost-Saving Strategies

```python
# Use fast models for simple tasks
> model:config simple_query gpt-4o-mini
> model:config classification claude-3-haiku

# Use Perplexity for research instead of GPT-4
> model:config research perplexity-sonar

# Monitor usage
> model:info
```

### Estimate Costs

```bash
# Estimate cost for different models
python howie_enhanced.py estimate-cost 1000 500

# Estimate for specific model
python howie_enhanced.py estimate-cost 1000 500 --model claude-3-opus
```

## Configuration Files

### Create Custom Configuration

```bash
# Interactive configuration
python howie_enhanced.py configure

# Save to custom location
python howie_enhanced.py configure --save my_models.json
```

### Example Configuration

```json
{
  "default_model": "gpt-4o",
  "task_mappings": {
    "research": "perplexity-sonar-pro",
    "analysis": "gpt-4o",
    "code_generation": "claude-3-5-sonnet",
    "simple_query": "gpt-4o-mini"
  },
  "agent_preferences": {
    "research": "perplexity-sonar-pro",
    "analysis": "gpt-4o",
    "code": "claude-3-5-sonnet",
    "optimization": "gpt-4o"
  }
}
```

## Advanced Usage

### Model-Specific Features

```python
# Vision models for analyzing images
> @gpt-4o analyze this screenshot of my league standings

# Research with real-time data
> @perplexity-sonar What happened in today's NFL games?

# Code generation with Claude
> @claude-3-5-sonnet Create a comprehensive analysis dashboard
```

### Parallel Agents with Different Models

```python
# Each agent can use its optimal model
> Run these tasks in parallel:
  - Research injury updates (uses Perplexity)
  - Analyze my roster (uses GPT-4o)  
  - Generate trade analysis code (uses Claude Sonnet)
  - Optimize my lineup (uses GPT-4o)
```

### Workflow Optimization

```python
# Complex workflow using multiple models
> Execute workflow:
  1. Research current player news (Perplexity)
  2. Analyze impact on my roster (GPT-4o)
  3. Generate optimization code (Claude Sonnet)
  4. Create comprehensive report (GPT-4o)
```

## CLI Commands

### Model Management

```bash
# List all models with costs
python howie_enhanced.py models

# Interactive configuration
python howie_enhanced.py configure

# Cost estimation
python howie_enhanced.py estimate-cost 1000 500

# Spawn agent with specific model
python howie_enhanced.py spawn "research playoff schedules" --model perplexity-sonar --agent research
```

### Chat Mode Commands

```python
# Model information
> model:info          # Show all models and usage
> model:switch <name> # Switch default model  
> model:config <task> <model> # Configure task mapping
> model:save         # Save configuration

# Model override
> @model_name <query> # Use specific model for one query

# Help
> help               # Show all commands including model commands
```

## Best Practices

### 1. **Choose the Right Model for the Task**
- Use **Perplexity** for research and current events
- Use **Claude Sonnet** for code generation
- Use **GPT-4o** for complex analysis
- Use **mini models** for simple queries

### 2. **Monitor Costs**
```python
> model:info  # Check usage and costs regularly
```

### 3. **Configure Task Mappings**
```python
# Set up your preferred models for each task type
> model:config research perplexity-sonar-pro
> model:config code_generation claude-3-5-sonnet
> model:save
```

### 4. **Use Model Overrides When Needed**
```python
# Force expensive model for important analysis
> @claude-3-opus Comprehensive playoff strategy analysis

# Force cheap model for simple tasks  
> @gpt-4o-mini List my bench players
```

## Troubleshooting

### API Key Issues
```bash
# Check which API keys are configured
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export PERPLEXITY_API_KEY="pplx-..."
```

### Model Not Available
```python
> model:info  # Check which models are properly configured
```

### Cost Concerns
```python
> model:info  # Check current usage and costs
# Configure cheaper models for routine tasks
> model:config simple_query gpt-4o-mini
```

### Performance Issues
```python
# Use faster models for planning and simple tasks
> model:config classification claude-3-haiku
```

## Integration with Existing Features

All existing Howie features work with the multi-model system:

- **Database queries** can use different models for analysis
- **Visualizations** can be created with cost-effective models
- **Code generation** uses the best coding models
- **Real-time data** can use research-optimized models
- **Agent spawning** uses optimal models per agent type

The multi-model system is fully backward compatible - existing workflows continue to work while gaining automatic optimization benefits!

## Summary

The multi-model system gives you:

1. **Automatic Optimization** - Right model for each task
2. **Cost Control** - Use expensive models only when needed
3. **Performance** - Fast models for simple tasks
4. **Flexibility** - Override model selection when needed
5. **Transparency** - Track usage and costs

This transforms Howie from using a single model to being a intelligent model orchestrator, similar to how Claude can access different models based on the task requirements!