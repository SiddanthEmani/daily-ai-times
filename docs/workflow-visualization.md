# GitHub Actions Workflow Visualization: collect-news.yml

## Workflow Overview

```mermaid
graph TD
    A[Workflow Triggers] --> B{Trigger Type}
    B -->|Schedule| C["🕐 Cron: Every 4 hours<br/>(0 */4 * * *)"]
    B -->|Manual| D["🔄 Workflow Dispatch<br/>- sources (optional)<br/>- force_refresh (boolean)<br/>- debug (boolean)"]
    
    C --> E[Collect Job]
    D --> E
    
    E --> F{Collect Success?}
    F -->|Yes| G[Deploy Job]
    F -->|No| H[❌ Workflow Failed]
    
    G --> I[✅ Deployment Complete]

    style A fill:#e1f5fe
    style E fill:#f3e5f5
    style G fill:#e8f5e8
    style H fill:#ffebee
    style I fill:#e8f5e8
```

## Detailed Job Flow

```mermaid
graph TD
    subgraph "🔧 COLLECT JOB"
        direction TB
        
        A1[🔄 Checkout Repository<br/>fetch-depth: 0]
        A2[🐍 Setup Python 3.11<br/>with pip cache]
        A3[📦 Cache System Dependencies<br/>/var/cache/apt]
        A4[⚙️ Install System Dependencies<br/>jq, rsync]
        A5[📚 Install Python Dependencies<br/>requirements.txt]
        
        A6[📰 Collect News<br/>collect_news.py]
        A7{News Collection<br/>Success?}
        A8[🔧 Generate API Files<br/>generate_api.py]
        A9[✅ Verify Generated Files<br/>Check JSON structure]
        
        A10[📝 Commit Changes<br/>Auto-commit with bot user]
        A11{Has Changes?}
        A12[📤 Push to Repository]
        A13[💾 Cache Deployment Files]
        
        A14[📁 Prepare Deployment Directory<br/>rsync frontend & API files]
        A15[🔍 Handle Google Analytics<br/>Injection or removal]
        A16[✅ Validate Deployment<br/>File count & structure check]
        A17[📤 Upload Pages Artifact]
        
        A1 --> A2 --> A3 --> A4 --> A5 --> A6
        A6 --> A7
        A7 -->|Success| A8
        A7 -->|Failure| Z1[❌ Stop Workflow]
        A8 --> A9 --> A10 --> A11
        A11 -->|Yes| A12
        A11 -->|No| A13
        A12 --> A13 --> A14 --> A15 --> A16 --> A17
    end
    
    subgraph "🚀 DEPLOY JOB"
        direction TB
        B1[🌐 Deploy to GitHub Pages<br/>deploy-pages@v4]
        B2[📊 Generate Workflow Summary<br/>Success/failure report]
        
        B1 --> B2
    end
    
    A17 --> B1
    
    style A6 fill:#fff3e0
    style A8 fill:#fff3e0
    style A14 fill:#e3f2fd
    style B1 fill:#e8f5e8
```

## Configuration & Permissions

```mermaid
graph LR
    subgraph "🔐 Permissions"
        P1[contents: write]
        P2[pages: write] 
        P3[id-token: write]
    end
    
    subgraph "⚙️ Job Configuration"
        C1[runs-on: ubuntu-latest]
        C2[timeout: 30 minutes]
        C3[concurrency: news-collection]
        C4[cancel-in-progress: false]
    end
    
    subgraph "🌍 Environment Variables"
        E1[PYTHONUNBUFFERED: 1]
        E2[PYTHONIOENCODING: utf-8]
    end
    
    style P1 fill:#ffebee
    style P2 fill:#ffebee
    style P3 fill:#ffebee
    style C1 fill:#e3f2fd
    style C2 fill:#e3f2fd
    style C3 fill:#e3f2fd
    style C4 fill:#e3f2fd
```

## File Flow & Data Pipeline

```mermaid
graph LR
    subgraph "📥 Input Sources"
        I1[sources.json<br/>Configuration]
        I2[User Inputs<br/>sources, force_refresh, debug]
        I3[Existing Data<br/>news.json archive]
    end
    
    subgraph "🔄 Processing"
        P1[collect_news.py<br/>Scrape & Process]
        P2[generate_api.py<br/>Create API endpoints]
    end
    
    subgraph "📁 Generated Files"
        O1[news.json<br/>Main data file]
        O2[latest.json<br/>Recent articles]
        O3[archives.json<br/>Historical data]
        O4[widget.json<br/>Widget data]
        O5[categories/*.json<br/>Categorized data]
    end
    
    subgraph "🌐 Deployment"
        D1[deploy_temp/<br/>Staging directory]
        D2[GitHub Pages<br/>Live website]
    end
    
    I1 --> P1
    I2 --> P1
    I3 --> P1
    P1 --> O1
    O1 --> P2
    P2 --> O2
    P2 --> O3
    P2 --> O4
    P2 --> O5
    
    O1 --> D1
    O2 --> D1
    O3 --> D1
    O4 --> D1
    O5 --> D1
    D1 --> D2
    
    style P1 fill:#fff3e0
    style P2 fill:#fff3e0
    style D2 fill:#e8f5e8
```

## Caching Strategy

```mermaid
graph TD
    subgraph "📦 Cache Layers"
        C1["System Dependencies<br/>/var/cache/apt<br/>Key: apt-cache-runner.os-workflow-hash"]
        C2["Python Dependencies<br/>pip cache<br/>Based on requirements.txt"]
        C3["Deployment Files<br/>frontend + API files<br/>Key: deployment-files-file-hashes"]
    end
    
    subgraph "🔄 Cache Usage"
        U1[Speeds up system package installation]
        U2[Avoids re-downloading Python packages]
        U3[Reuses unchanged deployment files]
    end
    
    C1 --> U1
    C2 --> U2
    C3 --> U3
    
    style C1 fill:#f3e5f5
    style C2 fill:#f3e5f5
    style C3 fill:#f3e5f5
```

## Error Handling & Validation

```mermaid
graph TD
    subgraph "🔍 Validation Steps"
        V1[Check news.json exists<br/>Validate article count]
        V2[Verify API files generated<br/>Count JSON files]
        V3[Validate index.html exists<br/>in deployment directory]
        V4[Clean up unwanted files<br/>README*.md removal]
    end
    
    subgraph "🚨 Error Scenarios"
        E1[News collection fails<br/>→ Stop workflow]
        E2[API generation fails<br/>→ Skip deployment steps]
        E3[Missing index.html<br/>→ Show directory structure & exit]
        E4[Analytics script not executable<br/>→ Skip injection, continue]
    end
    
    V1 --> E1
    V2 --> E2
    V3 --> E3
    V4 --> E4
    
    style E1 fill:#ffebee
    style E2 fill:#ffebee
    style E3 fill:#ffebee
    style E4 fill:#fff3e0
```

## Workflow Summary Features

- **🕐 Automated Schedule**: Runs every 4 hours automatically
- **🔄 Manual Trigger**: Can be run manually with custom parameters
- **📊 Smart Deployment**: Only deploys when there are actual changes
- **⚡ Performance Optimized**: Uses caching and rsync for efficiency
- **🔍 Debug Mode**: Optional detailed logging for troubleshooting
- **📱 Google Analytics**: Automatic injection or removal based on secrets
- **🤖 Auto-commit**: Commits changes with bot user credentials
- **📈 Workflow Reports**: Generates summary of execution results
