# main.py

import argparse
from train import train
from evaluate import evaluate

def main():
    parser = argparse.ArgumentParser(description="AUV Path Planning using RL")
    parser.add_argument("--mode", choices=["train", "eval"], required=True, help="Mode: train or eval")
    parser.add_argument("--model_path", type=str, default="auv_sac_model", help="Path to the trained model (for evaluation)")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true", help="Print step-by-step info")

    args = parser.parse_args()

    if args.mode == "train":
        train()
    elif args.mode == "eval":
        evaluate(model_path=args.model_path, episodes=args.episodes, render=args.render)

if __name__ == "__main__":
    main()
