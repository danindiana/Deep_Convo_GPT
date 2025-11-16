library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity BrainModel is
    Port ( Clk : in  STD_LOGIC;
           Input : in  STD_LOGIC_VECTOR (7 downto 0);
           Store : in  STD_LOGIC;
           Retrieve : in  STD_LOGIC;
           Output : out  STD_LOGIC_VECTOR (7 downto 0));
end BrainModel;

architecture Behavior of BrainModel is
    signal memory : STD_LOGIC_VECTOR (7 downto 0);
begin
    process (Clk)
    begin
        if rising_edge(Clk) then
            if Store = '1' then
                memory <= Input;
            end if;
            if Retrieve = '1' then
                Output <= memory;
            end if;
        end if;
    end process;
end Behavior;
