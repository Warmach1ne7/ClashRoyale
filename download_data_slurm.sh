#!/bin/bash
#SBATCH -N 1
#SBATCH -n 10
#SBATCH --mem=50g
#SBATCH -J "clash_royale_placement"
#SBATCH -p short
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:1

# Load CUDA modules
module load cuda12.6/toolkit
module load cuda12.6/blas
module load cuda12.6/fft

/home/ostikar/.conda/envs/clashroyale/bin/python download_data.py --game 86643da7-dbed-4e38-a61b-dc21883e6600
/home/ostikar/.conda/envs/clashroyale/bin/python download_data.py --game 913d968b-867a-47e1-b472-1364cbb5a511
/home/ostikar/.conda/envs/clashroyale/bin/python download_data.py --game 96890afe-0169-4741-89bf-9c005df33353
/home/ostikar/.conda/envs/clashroyale/bin/python download_data.py --game a86d728c-8f39-4d31-af05-d679fcdf6233
/home/ostikar/.conda/envs/clashroyale/bin/python download_data.py --game a907aca7-2aac-4439-b5fc-5fdac4cb72b1
/home/ostikar/.conda/envs/clashroyale/bin/python download_data.py --game aa29e2a1-9dec-4678-bf9b-2ec216d67c43
/home/ostikar/.conda/envs/clashroyale/bin/python download_data.py --game b0bd4934-337c-4e50-bbe9-ec3551af96a0
/home/ostikar/.conda/envs/clashroyale/bin/python download_data.py --game b4b4602c-ab60-4d90-844d-c9012a0ba760
/home/ostikar/.conda/envs/clashroyale/bin/python download_data.py --game c0e32272-f938-409f-b9d1-ae6f43d49603
/home/ostikar/.conda/envs/clashroyale/bin/python download_data.py --game ca371685-d5d0-4910-a6c2-3a19e692d201
/home/ostikar/.conda/envs/clashroyale/bin/python download_data.py --game cf481ce1-e68a-4102-9cc5-402589f5d552
/home/ostikar/.conda/envs/clashroyale/bin/python download_data.py --game ee4a7db6-d938-4d50-ad74-9dbee28bd5f0
/home/ostikar/.conda/envs/clashroyale/bin/python download_data.py --game f2142638-0107-4965-a17f-8685e1f1cb82
/home/ostikar/.conda/envs/clashroyale/bin/python download_data.py --game f6204b05-0353-45a2-958f-d83fd9045170

/home/ostikar/.conda/envs/clashroyale/bin/python download_data.py --game 1ab0e0aa-1323-41e1-93e4-5cc2307c2913
/home/ostikar/.conda/envs/clashroyale/bin/python download_data.py --game 2816eac0-552f-4ac8-b26b-5d28ab2ee862
/home/ostikar/.conda/envs/clashroyale/bin/python download_data.py --game 2f371c8c-d2bd-4434-a89a-939582595d43
/home/ostikar/.conda/envs/clashroyale/bin/python download_data.py --game 341e6b4a-495a-4d73-9f69-8a15fca347e3
/home/ostikar/.conda/envs/clashroyale/bin/python download_data.py --game 4967ee6f-56e2-41d2-84d2-4e358945b9b0
/home/ostikar/.conda/envs/clashroyale/bin/python download_data.py --game 5bb37fd4-52a2-4ff6-8ad1-0ff418aa9082
/home/ostikar/.conda/envs/clashroyale/bin/python download_data.py --game 5dab889d-26de-49d9-acb1-04e7fd119da0
/home/ostikar/.conda/envs/clashroyale/bin/python download_data.py --game 5f986463-0a5a-4f6b-9e08-c34382c66fc0
/home/ostikar/.conda/envs/clashroyale/bin/python download_data.py --game 62e57a10-7f32-4acf-a3c8-62abe1ad5723
/home/ostikar/.conda/envs/clashroyale/bin/python download_data.py --game 6444504a-7dd8-4add-bc69-43a489ebc531
/home/ostikar/.conda/envs/clashroyale/bin/python download_data.py --game 67daa821-e0b6-480f-9149-b914975c7410
/home/ostikar/.conda/envs/clashroyale/bin/python download_data.py --game 7478b10f-0571-49f4-a3c4-c0b21647b651
/home/ostikar/.conda/envs/clashroyale/bin/python download_data.py --game 776cf365-756f-467e-a14a-cd4414702b00
/home/ostikar/.conda/envs/clashroyale/bin/python download_data.py --game 79aa5c0c-a736-446a-8fe0-98642c676c92
/home/ostikar/.conda/envs/clashroyale/bin/python download_data.py --game 7eeac525-963c-4d64-8eeb-aca80c558553
/home/ostikar/.conda/envs/clashroyale/bin/python download_data.py --game 8495e602-3d89-4057-b096-29ba9e0810c1
/home/ostikar/.conda/envs/clashroyale/bin/python download_data.py --game 88ad5632-0640-4e6b-89c8-b54e0b8c8b31
/home/ostikar/.conda/envs/clashroyale/bin/python download_data.py --game 88fe1c67-8d1e-4a31-9485-89ae67119ea0
/home/ostikar/.conda/envs/clashroyale/bin/python download_data.py --game 8b083411-4fd4-4176-9475-85e682d3afd1
/home/ostikar/.conda/envs/clashroyale/bin/python download_data.py --game 96445ca8-2020-416d-8c82-6ddf36aeedc1
/home/ostikar/.conda/envs/clashroyale/bin/python download_data.py --game 9bf20394-521c-4c83-a3e6-dde9d7301db2
/home/ostikar/.conda/envs/clashroyale/bin/python download_data.py --game a5918494-7c12-4f57-bfd2-c990180114e1
/home/ostikar/.conda/envs/clashroyale/bin/python download_data.py --game b058d5d1-57ab-4c92-81d4-55b8dbeb08b2
/home/ostikar/.conda/envs/clashroyale/bin/python download_data.py --game b05988a2-1a75-4ad1-9e27-47eb519e1d43
/home/ostikar/.conda/envs/clashroyale/bin/python download_data.py --game cc538bd8-7131-4629-8ef2-da15f34b6812
/home/ostikar/.conda/envs/clashroyale/bin/python download_data.py --game d299629a-ab6a-4db9-8fc4-089621723b31
/home/ostikar/.conda/envs/clashroyale/bin/python download_data.py --game d7a2d172-a92b-4e0e-a3b2-de614e0df963
/home/ostikar/.conda/envs/clashroyale/bin/python download_data.py --game e80b9e94-23e5-4d70-9408-297b7ddb0df1
/home/ostikar/.conda/envs/clashroyale/bin/python download_data.py --game eb93d114-9b3e-434b-a5c8-a251feba2c03
/home/ostikar/.conda/envs/clashroyale/bin/python download_data.py --game ef6bf415-ac26-4ebf-ac00-a79ae9cc65f2
/home/ostikar/.conda/envs/clashroyale/bin/python download_data.py --game f21c5dc0-5614-4491-8eeb-9f6377086860
