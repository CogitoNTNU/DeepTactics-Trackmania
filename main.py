
run_tm = True

if run_tm:
    from src.env_tm import run_training
    print("\n" + "="*80)
    print("LOADING TRACKMANIA ENVIRONMENT (env_tm)")
    print("="*80 + "\n")
else:
    from src.env import run_training
    print("\n" + "="*80)
    print("LOADING GYMNASIUM ENVIRONMENT (env)")
    print("="*80 + "\n")

if __name__ == "__main__":
    run_training()